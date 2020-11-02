"""Stock Dataset class.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config

import ast
import numpy as np 
import os
import pandas as pd 
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
        hgs, node_embs = fetch_data(today)  # hgs: (num_days, 2, x)  node_embs: (num_days, stock_num, 768)
        return hgs, node_embs, y, prices
        
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
    for idx in range(idx-config.LOOKBACK_WINDOW, idx+1):
        dates.append(DATES[idx])

    return dates

def fetch_data(today):
    '''Returns hgs, node_embs given today's date.
    @param   today     (str)                : Today's date.
    @returns hgs       (List[torch.tensor]) : List of hypergraphs.      tensor.shape: (num_days, 2, x)
    @returns node_embs (List[torch.tensor]) : List of node embeddings.  tensor.shape: (num_days, num_stocks, 768)
    '''
    hgs = []
    node_embs = []  # doc_embs
    dates = lookback_window_dates(today)
    num_edges = 0
    for date in dates:
        hg = np.load(config.HG_PATH + date + ".npy")
        num_edges += hg.shape[1]
        #* Process the npy hg to feed it to the hgnn
        inci_sparse = sparse.coo_matrix(hg)
        incidence_edge = utils.from_scipy_sparse_matrix(inci_sparse)
        hyp_input = incidence_edge[0] # this is edge list (2, x)

        #* Fetch node_emb
        node_emb = []
        for article in sorted(os.listdir(os.path.join(config.ARTICLES, date))):
            a = torch.load(os.path.join(config.ARTICLES, date, article), map_location='cpu')
            node_emb.append(a[:, :config.BERT_SIZE])

        hgs.append(hyp_input)
        node_embs.append(node_emb)
    return hgs, node_embs
    
def node_emb_generate(date):
    '''Returns node_emb for a given date.
    @param   date     (str)          : Date for node_emb. 
    @returns node_emb (torch.tensor) : Node embedding.     tensor.shape: (num_stocks, 768)   
    '''
    NAMES_HG = open(config.NAMES_HG_PATH, "r")
    DATES = sorted(open(config.DATES_PATH, "r").read().split())
    TICKERS = sorted(open(config.TICKERS_PATH, "r").read().split())
    tick_to_idx = {}  # ticker to index
    idx_to_tick = {}  # index to ticker  # unused

    for idx, ticker in enumerate(TICKERS):
        tick_to_idx[ticker] = idx
        idx_to_tick[idx] = ticker

    stock_embs = torch.load(config.STOCK_EMB_PATH, map_location="cpu")  # (num_stocks, 768)
    node_emb = torch.zeros(len(TICKERS), 768)  # (num_stocks, 768)

    for reports, day in zip(NAMES_HG, DATES):
        if(day == date):
            reports = list(ast.literal_eval(reports))
            print(reports)
            # for report in reports:
            #     for stock in report:
            #         node_emb[tick_to_idx[stock]] += stock_embs[tick_to_idx[stock]]
            break

    return node_emb