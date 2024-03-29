"""Top-level model class.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import attention
import config
import dataset

import numpy as np 
import time
import torch 
import torch_geometric
import torch.nn as nn 
import torch.nn.functional as F 

class StockModel(nn.Module): # TODO hgnn, lstm, fc details
    def __init__(self, stock_emb_dim=config.args.BERT_SIZE, hidden_size=16, heads=4, negative_slope=0.2, dropout=0.1):
        """Init StockModel.
        @param stock_emb_dim  (int)  : BERT stock emb dimension.
        @param hidden_size    (int)  : Hidden dimension size.
        @param heads          (int)  : Number of heads for HypergraphConv.
        @param negative_slope (float): Negative_slope for HypergraphConv.
        @param dropout        (float): Dropout probability.
        """
        super(StockModel, self).__init__()
        self.hidden_size = hidden_size
        self.price_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, bias=True, batch_first=False)
        self.lstm = nn.LSTM(input_size=hidden_size+config.args.BERT_SIZE, hidden_size=hidden_size, bias=True, batch_first=False)
        self.fc1 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, con_e_list, adj_u_list, article_embs, prices):
        """ Takes a lookback window number of hypergraphs, node embeddings for those hypergraphs 
        and prices in the lookback window days and predicts if stock price of all tickers 
        increase(1) or decrease(0) the next day.

        @param hgs       (List[torch.tensor]): List of hypergraphs.       tensor.shape: (1, 2, x)
        @param node_embs (List[torch.tensor]): List of node embeddings.   tensor.shape: (1, stock_num, 768)
        @param prices    (List[torch.tensor]): List of prices.            tensor.shape: (1, stock_num, 1)

        Length of all lists: 4 (size of the lookback window)

        @returns out     (torch.tensor): Prediction of all tickers.       tensor.shape: (stock_num, 2)
        """
        # start = time.time()
        # START = time.time()
        ###* Price embedding for node:
        ###* Using all the hidden states of the lstm
        prices = torch.cat(prices, dim=0)  # (num_days, stock_num, 1)
        new_prices, (h_n, c_n) = self.price_lstm(prices.to(config.DEVICE))  # (num_days, stock_num, self.hidden_size)
        # print("price emb for node: ", round(time.time()-start, 5))
        # start = time.time()
        # temp = time.time()
        ###* DHGNN:
        hg_outputs = []
        for con_e_, adj_u_, article_emb, price_emb in zip(con_e_list, adj_u_list, article_embs, new_prices):
            con_e = {}
            adj_u = {}
            for k, v in con_e_.items():
                con_e[k] = [int(ele) for ele in v]
            for k, v in adj_u_.items():
                adj_u[k] = [int(ele) for ele in v]
            ###* VertexConv followed by EdgeConv for every vertex in adj_u
            hg_tensor = torch.zeros(116, config.args.BERT_SIZE+self.hidden_size).cuda()
            for vertex, hyper_edge_set in adj_u.items():
                ###* VertexConv of all vertices for every hyper edge, followed by concat with article emb
                hyper_edge_emb_list = []
                for hyper_edge in hyper_edge_set:
                    he_node_embs = price_emb[con_e[int(hyper_edge)]]
                    vc = attention.VertexConv(dim_in=he_node_embs.shape[1], k=he_node_embs.shape[0]).cuda()  # (dim_in = 32, k = number of vertices)
                    he_emb = vc(he_node_embs.unsqueeze(0).cuda())
                    he_emb_cat = torch.cat((he_emb, article_emb[hyper_edge].squeeze(0).cuda()), dim=-1)
                    hyper_edge_emb_list.append(he_emb_cat.view(he_emb_cat.shape[0], 1, he_emb_cat.shape[1])) 
                hyper_edge_emb_list = torch.cat(hyper_edge_emb_list, dim=1)
                ###* Edge conv over the concatenated hyperedge embeddings
                ec = attention.EdgeConv(hyper_edge_emb_list.shape[-1], hyper_edge_emb_list.shape[-1]//4).cuda()
                ec_out = ec(hyper_edge_emb_list.float())
                ###* Storing this embedding for the corresponding vertex:
                hg_tensor[vertex] += ec_out.view(-1)
            hg_outputs.append(hg_tensor)
            # print("vertexconv -> edgeconv: ", round(time.time()-temp, 5))
            # temp = time.time()
            # print()
        # print("dhgnn: ", round(time.time() - start, 5))
        # start = time.time()
        ###* Passing the output from HGNNs into a LSTM:
        ###* Using all the hidden states of the lstm
        hg_outputs = torch.cat(hg_outputs).view(-1, config.STOCK_NUM, self.hidden_size+config.args.BERT_SIZE)  # (num_days, stock_num, hidden_size + bert_size) = (4, 116, 800)
        lstm_out, (h_n_o, c_n_o) = self.lstm(hg_outputs)  # (num_days, stock_num, hidden_size)
        ###* Skip connection:
        lstm_attention = lstm_out + new_prices
        # print("lstm -> skip: ", round(time.time()-start, 5))
        # start = time.time()
        ###* Self attention over lstm outputs:
        query_a = lstm_attention[-1].view(1, lstm_attention.shape[1], lstm_attention.shape[2])  # last day embedding
        query_a = query_a.permute(1, 0, 2)  # (stock_num, 1, hidden_size)
        context_a = lstm_attention.permute(1, 0, 2)  # (stock_num, 4, hidden_size)
        attention_out, _ = attention.Attention(dimensions=self.hidden_size).cuda()(query=query_a, context=context_a)  # (stock_num, 1, hidden_size)
        attention_out = attention_out.squeeze(1)  # (stock_num, hidden_size)
        ###* Linear:
        out = self.dropout(self.fc1(attention_out))  # (stock_num, 2)
        # print("attention -> linear: ", round(time.time()-start, 5))
        # print("\nTOTAL TIME: ", time.time()-START)
        return out