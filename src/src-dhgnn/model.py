"""Top-level model class.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config
import dataset


import numpy as np 
import torch 
import torch_geometric
import torch.nn as nn 
import torch.nn.functional as F 

class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    Using self-attention to aggregate hyperedges
    """
    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super(EdgeConv, self).__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        scores = []
        n_edges = ft.size(1)
        for i in range(n_edges):
            scores.append(self.fc(ft[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        
        return (scores * ft).sum(1)


class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super(Transform, self).__init__()

        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats.cuda())  # (N, k*k, 1)
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier)  # softmax along last dimension
        transformed_feats = torch.matmul(multiplier, region_feats.cuda())  # (N, k, d)
        return transformed_feats


class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super(VertexConv, self).__init__()

        self.trans = Transform(dim_in, k)                   # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)                    # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        transformed_feats = self.trans(region_feats.cuda())
        pooled_feats = self.convK1(transformed_feats)             # (N, 1, d)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats


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
        self.price_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, bias=True, batch_first=False)
        self.lstm = nn.LSTM(input_size=hidden_size+768, hidden_size=hidden_size, bias=True, batch_first=False)
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
        
        new_prices = []
        prices = torch.cat(prices, dim=0)
        # print("__"*80)
        # print("price shape:")
        # print(prices.shape)
        # new_prices = 
        output, (h_n, c_n) = self.price_lstm(prices.to(config.DEVICE))
        # print(h_n.shape, c_n.shape, output.shape)
        for idx in range(output.shape[0]):
            new_prices.append(output[idx])
        # print(new_prices[0].shape, len(new_prices))

        # print("__"*80)
        ### DHGNN:
        hg_outputs = []
        for hg, node_emb, price_emb in zip(hgs, node_embs, new_prices):
            con_e = {}
            adj_u = {}

            hg_tensor = torch.zeros(116, 800).cuda()
            # print("price shape")
            # print(price.shape)

            # price_emb = self.price_emb(price.squeeze(0).to(config.DEVICE))

            hg = hg.squeeze(0).to(config.DEVICE)
            # hg = node_emb.squeeze(0)

            hyper_edges_id = sorted(list(set(hg[1].cpu().numpy().tolist())))
            for ele in set(hyper_edges_id):
                ids = np.where(hg[1].cpu().numpy() == ele)[0]
                con_e[ele] = hg[0][ids].cpu().numpy().tolist()
            
            vertices_id = sorted(list(set(hg[0].cpu().numpy().tolist())))
            for ele in set(vertices_id):
                ids = np.where(hg[0].cpu().numpy() == ele)[0]
                adj_u[ele] = hg[1][ids].cpu().numpy().tolist()
            
            # print(con_e)
            # print()
            # print(adj_u)
            # print()
            for vertex, hyper_edge_set in adj_u.items():
                hyper_edge_list = []
                # print("__"*80)
                # print(vertex, hyper_edge_set)
                for hyper_edge in hyper_edge_set:
                    a = price_emb[con_e[hyper_edge]]
                    # print("A shape:")
                    # print(a.shape)  # (k, 32)
                    # b = node_emb[hyper_edge]
                    vc = VertexConv(dim_in=a.shape[1], k=a.shape[0]).cuda()  # (dim_in = 32, k = number of vertices)
                    a_b = vc(a.unsqueeze(0).cuda())
                    # print("a_b shape:")
                    # print(a_b.shape)  # (1, 32)
                    # print("hyper_Edge:")
                    # print(hyper_edge)
                    # print("node_emb[hyperedge].shape")
                    # print(node_emb[hyper_edge].shape)
                    z = torch.cat((a_b, node_emb[hyper_edge].squeeze(0).cuda()), dim=-1)
                    # print("z.shape")
                    # print(z.shape)
                    hyper_edge_list.append(z.view(z.shape[0], 1, z.shape[1]))
                    # hyper_edge_list.append(z)
                    # print("__"*80)

                    # print(a.shape, b.shape)
                # print(hyper_edge_list)
                hyper_edge_list = torch.cat(hyper_edge_list, dim=1)
                # print("hyper_edge_list:")
                # print(hyper_edge_list.shape)

                ec = EdgeConv(hyper_edge_list.shape[-1], hyper_edge_list.shape[-1]//4).cuda()
                ec_out = ec(hyper_edge_list)
                # print("ec_out:")
                # print(ec_out.shape)
                hg_tensor[vertex] += ec_out.view(-1)
                # for ele in hyper_edge_list:
                #     print(ele.shape)
                # print("hg_tensor:")
                # print(hg_tensor.shape, vertex)
                # print("__"*80)
                # print(hg_tensor[0])
                # print("__"*80)
                # print(hg_tensor[vertex])
                # print("__"*80)
                # break
            hg_outputs.append(hg_tensor)
            # break
        # print("hg_outputs:")
        # print(len(hg_outputs), hg_outputs[0].shape)



        ### Passing the output from HGNNs into a LSTM followed by linear layers.
        hg_outputs = torch.cat(hg_outputs).view(-1, config.STOCK_NUM, self.hidden_size+768)  # (num_days, stock_num, hidden_size+768) = (4, 116, 800)
        lstm_out, _ = self.lstm(hg_outputs)[-1]  # (1, stock_num, hidden_size)
        lstm_out = lstm_out.squeeze(0)  # (stock_num, hidden_size)
        out = self.fc2(self.dropout(self.fc1(lstm_out)))  # (stock_num, 2)
        
        return out
        
        # return torch.randn(116, 2)
