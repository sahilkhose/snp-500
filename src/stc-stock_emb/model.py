"""Top-level model class.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config
import dataset


import ast
import os
import torch 
import torch_geometric
import torch.nn as nn 
import torch.nn.functional as F 


class StockModel(nn.Module): # TODO hgnn, lstm, fc details
    def __init__(self, stock_emb_dim=768, hidden_size=32, heads=4, negative_slope=0.2, dropout=0.1):
        """Init StockModel.
        
        @param stock_emb_dim  (int)  : BERT stock emb dimension.
        @param hidden_size    (int)  : Hidden dimension size.
        @param heads          (int)  : Number of heads for HypergraphConv.
        @param negative_slope (float): Negative_slope for HypergraphConv.
        @param dropout        (float): Dropout probability.
        """
        super(StockModel, self).__init__()
        self.hidden_size = hidden_size
        self.price_emb = nn.Linear(1, hidden_size)
        self.hgnn1 = torch_geometric.nn.HypergraphConv(
            stock_emb_dim + hidden_size, 
            hidden_size, 
            use_attention=True, 
            heads=heads, 
            concat=False, 
            negative_slope=negative_slope,
            dropout=0.3, 
            bias=True
            )
        self.hgnn2 = torch_geometric.nn.HypergraphConv(
            hidden_size, 
            hidden_size, 
            use_attention=True, 
            heads=heads, 
            concat=False, 
            negative_slope=negative_slope,
            dropout=0.3, 
            bias=True
            )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bias=True, batch_first=False)
        self.fc1 = nn.Linear(hidden_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, 2)
        self.dropout = nn.Dropout(dropout)

        self.stock_emb = nn.Embedding(config.STOCK_NUM, 768)

    def create_market_vector(self, date):
        '''Returns node_emb for a given date.
        @param   date     (str)          : Date for node_emb. 
        @returns node_emb (torch.tensor) : Node embedding.     tensor.shape: (num_stocks, 768)   
        '''
        embs_day = os.path.join(config.ARTICLES_EMB, date)

        ### score(i, j) = n_key(i) . s(j)    i: articles, j:stocks
        score = torch.zeros(len(os.listdir(embs_day)), config.STOCK_NUM).cuda()  # (num_articles, num_stocks)

        for article_id, article in enumerate(sorted(os.listdir(embs_day))):
            n_key = torch.load(os.path.join(embs_day, article))  # (1, 768)
            a = torch.zeros(config.STOCK_NUM).cuda()  # (num_stocks)
            for i in range(config.STOCK_NUM):
                a[i] += torch.dot(
                    n_key.view(-1), 
                    self.stock_emb(torch.LongTensor([i]).cuda()).cuda().view(-1)
                    )                
            score[article_id] += a

        ### alpha(i, j) = softmax(score(i, j)) over i (articles)
        alpha = nn.Softmax(dim=0)(score).cuda()  # (2, num_stocks)

        ### market_vector(j) = sum(alpha(i, j) . n_value(i))
        market_vector = torch.zeros(config.STOCK_NUM, 768).cuda()  # (num_stocks, 768)

        for article_id, article in enumerate(sorted(os.listdir(embs_day))):
            n_key = torch.load(os.path.join(embs_day, article))  # (1, 768)
            market_vector += alpha[article_id].view(-1, 1).cuda() * n_key
        return market_vector


    def forward(self, hgs, node_embs, prices, today):
        """ Takes a lookback window number of hypergraphs, node embeddings for those hypergraphs 
        and prices in the lookback window days and predicts if stock price of all tickers 
        increase(1) or decrease(0) the next day.

        @param hgs       (List[torch.tensor]): List of hypergraphs.       tensor.shape: (1, 2, x)
        @param node_embs (List[torch.tensor]): List of node embeddings.   tensor.shape: (1, stock_num, 768)
        @param prices    (List[torch.tensor]): List of prices.            tensor.shape: (1, stock_num, 1)
        @param today

        Length of all lists: 4 (size of the lookback window)

        @returns out     (torch.tensor): Prediction of all tickers.       tensor.shape: (stock_num, 2)
        """
        ### Combining price embeddings with node embeddings.
        # node_price_embs = []  # List to concat node_emb + price_emb

        dates = dataset.lookback_window_dates(today)
        market_vectors = []
        for date in dates:
            market_vectors.append(self.create_market_vector(date))

        market_vectors_embs = []  # List to concat market_vector + price_emb
        
        for price, node_emb in zip(prices, market_vectors):
            price_emb = self.price_emb(price.squeeze(0).to(config.DEVICE))  # (stock_num, hidden_size)
            price_emb = price_emb.view(1, config.STOCK_NUM, self.hidden_size)  # (1, stock_num, hidden_size)
            # print(price_emb.shape, node_emb.shape)
            market_vectors_embs.append(torch.cat((node_emb.to(config.DEVICE).view(1, config.STOCK_NUM, 768), price_emb), dim=2))  # (1, stock_num, 768 + hidden_size)

        ### Passing hypergraphs and combined node embeddings through HGNNs
        hg_outputs = []  # List to concat hgnn outputs for lstm

        for hg, node_price_emb in zip(hgs, market_vectors_embs):
            hgnn1 = self.hgnn1(node_price_emb.squeeze(0).to(config.DEVICE), hg.squeeze(0).to(config.DEVICE))  # (stock_num, hidden_size)
            hg_outputs.append(self.hgnn2(hgnn1.to(config.DEVICE), hg.squeeze(0).to(config.DEVICE)))  # (stock_num, hidden_size)

        ### Passing the output from HGNNs into a LSTM followed by linear layers.
        hg_outputs = torch.cat(hg_outputs).view(-1, config.STOCK_NUM, self.hidden_size) # (num_days, stock_num, hidden_size)
        lstm_out, _ = self.lstm(hg_outputs)[-1]  # (1, stock_num, hidden_size)
        lstm_out = lstm_out.squeeze(0)  # (stock_num, hidden_size)
        out = self.fc2(self.dropout(self.fc1(lstm_out)))  # (stock_num, 2)
        
        return out










if __name__ == "__main__":
    model = StockModel().cuda()
    print(model.create_market_vector("2006-10-20").shape)
    # hgs, node_embs = dataset.fetch_data("2013-11-20")
    # print("__"*80)
    # out = model(hgs, node_embs)
    # print(out.shape)
    # y = pd.read_csv("LABELS.csv", index_col=0)
    # y = torch.tensor(y.loc["2013-11-20", :], dtype=torch.long)
    # print(y.shape)
    # a = nn.CrossEntropyLoss()(out, y)
    # print(a)
    # [print(par) for par in model.parameters()]
    # print(model.parameters())