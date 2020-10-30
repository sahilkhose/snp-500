"""Top-level model class.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config
import dataset


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

    def forward(self, hgs, node_embs, prices):
        """ Takes a lookback window number of hypergraphs, node embeddings for those hypergraphs 
        and prices in the lookback window days and predicts if stock price of all tickers 
        increase(1) or decrease(0) the next day.

        @param hgs       (List[torch.tensor]): List of hypergraphs.       tensor.shape: (1, 2, x)
        @param node_embs (List[torch.tensor]): List of node embeddings.   tensor.shape: (1, stock_num, 768)
        @param prices    (List[torch.tensor]): List of prices.            tensor.shape: (1, stock_num, 1)

        Length of all lists: 4 (size of the lookback window)

        @returns out     (torch.tensor): Prediction of all tickers.       tensor.shape: (stock_num, 2)
        """
        ### Combining price embeddings with node embeddings.
        node_price_embs = []  # List to concat node_emb + price_emb
        
        for price, node_emb in zip(prices, node_embs):
            price_emb = self.price_emb(price.squeeze(0).to(config.DEVICE))  # (stock_num, hidden_size)
            price_emb = price_emb.view(1, config.STOCK_NUM, self.hidden_size)  # (1, stock_num, hidden_size)
            node_price_embs.append(torch.cat((node_emb.to(config.DEVICE), price_emb), dim=2))  # (1, stock_num, 768 + hidden_size)

        ### Passing hypergraphs and combined node embeddings through HGNNs
        hg_outputs = []  # List to concat hgnn outputs for lstm

        for hg, node_price_emb in zip(hgs, node_price_embs):
            hgnn1 = self.hgnn1(node_price_emb.squeeze(0).to(config.DEVICE), hg.squeeze(0).to(config.DEVICE))  # (stock_num, hidden_size)
            hg_outputs.append(self.hgnn2(hgnn1.to(config.DEVICE), hg.squeeze(0).to(config.DEVICE)))  # (stock_num, hidden_size)

        ### Passing the output from HGNNs into a LSTM followed by linear layers.
        hg_outputs = torch.cat(hg_outputs).view(-1, config.STOCK_NUM, self.hidden_size) # (num_days, stock_num, hidden_size)
        lstm_out, _ = self.lstm(hg_outputs)[-1]  # (1, stock_num, hidden_size)
        lstm_out = lstm_out.squeeze(0)  # (stock_num, hidden_size)
        out = self.fc2(self.dropout(self.fc1(lstm_out)))  # (stock_num, 2)
        
        return out










if __name__ == "__main__":
    model = StockModel()
    hgs, node_embs = dataset.fetch_data("2013-11-20")
    print("__"*80)
    out = model(hgs, node_embs)
    print(out.shape)
    y = pd.read_csv("LABELS.csv", index_col=0)
    y = torch.tensor(y.loc["2013-11-20", :], dtype=torch.long)
    print(y.shape)
    a = nn.CrossEntropyLoss()(out, y)
    print(a)
    # [print(par) for par in model.parameters()]
    # print(model.parameters())