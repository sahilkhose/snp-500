"""Stock Dataset class.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config

import ast
import numpy as np 
import os
import pandas as pd 
import pickle
import torch
import torch_geometric

from scipy import sparse
import time
from torch_geometric import utils


class StockDataset:
    def __init__(self, x, y):
        '''Init StockDataset

        @param x (List[str])    : List of dates.     len(List): num_days
        @param y (pd.DataFrame) : Labels dataframe.  shape    : (num_days, num_stocks)
        '''
        self.x = x  
        self.y = y  

    def __len__(self):
        '''Returns num_days
        '''
        return len(self.x)

    def __getitem__(self, index):
        '''Returns hgs, node_embs, y and prices in the lookback window given today's date index.
        @param   index     (int)                : Date index.
        @returns hgs       (List[torch.tensor]) : List of hypergraphs.      tensor.shape: (num_days, 2, x)
        @returns node_embs (List[torch.tensor]) : List of node embeddings.  tensor.shape: (num_days, num_stocks, 768)
        @returns y         (torch.tensor)       : Label.                    tensor.shape: (num_stocks)
        @returns prices    (List[torch.tensor]) : List of prices.           tensor.shape: (num_days, num_stocks, 1)
        '''
        #* Selecting today's date:
        today = str(self.x[index])

        #* Get label
        y = self.y.loc[today, :]  # today's label
        y = torch.tensor(y, dtype=torch.long)  # (num_stocks)

        #* Fetching prices
        prices = []
        dates = lookback_window_dates(today)  # list of dates
        price_df = pd.read_csv(config.PRICE_PATH, index_col=0)  # df for price info
        for date in dates:
            price = price_df.loc[date, :]  
            price = torch.tensor(price, dtype=torch.float).view(-1, 1)  # (num_stocks, 1)
            prices.append(price)

        #* Fetching hypergraphs and node embeddings
        con_e_list, adj_u_list, article_embs = fetch_data(today)  # hgs: (num_days, 2, x)  node_embs: (num_days, stock_num, 768)

        return con_e_list, adj_u_list, article_embs, y, prices
        
def lookback_window_dates(today):
    '''Returns dates in the lookback window given today's date.
    @param   today (str)       : Today's date.
    @returns dates (List(str)) : Dates in the lookback window along with today.
    '''
    dates = []
    DATES = sorted(open(config.DATES_PATH, "r").read().split())
    for idx, date_temp in enumerate(DATES):
        if date_temp == today:
            break
    for idx in range(idx-config.args.LOOKBACK_WINDOW, idx+1):
        dates.append(DATES[idx])

    return dates

def fetch_data(today):
    '''Returns hgs, node_embs given today's date.
    @param   today     (str)                : Today's date.
    @returns hgs       (List[torch.tensor]) : List of hypergraphs.      tensor.shape: (num_days, 2, x)
    @returns node_embs (List[torch.tensor]) : List of node embeddings.  tensor.shape: (num_days, num_stocks, 768)
    '''
    con_e_list = []
    adj_u_list = []
    article_embs = []
    dates = lookback_window_dates(today)
    for date in dates:
        #* Fetch con_e and adj_u dictionaries:
        confile = open(os.path.join(config.CON_E_PATH, date),'rb')
        con_e = pickle.load(confile)
        con_e_list.append(con_e)
        confile.close()

        adjfile = open(os.path.join(config.ADJ_U_PATH, date),'rb')
        adj_u = pickle.load(adjfile)
        adj_u_list.append(adj_u)
        adjfile.close()

        #* Fetch article_emb
        article_emb = []
        for article in sorted(os.listdir(os.path.join(config.ARTICLES, date))):
            a = torch.tensor(torch.load(os.path.join(config.ARTICLES, date, article), map_location='cpu'), dtype=float)
            # article_emb.append(a[:, :config.args.BERT_SIZE])
            article_emb.append(a.view(1, -1))
        article_embs.append(article_emb)
    return con_e_list, adj_u_list, article_embs