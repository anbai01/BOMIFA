import torch.nn.functional as F
import torch.nn as nn
import torch

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        x = x.to(self.weight.device)
        support = torch.mm(x, self.weight)
        adj = adj.to(self.weight.device)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class GCN_E(nn.Module):

    def __init__(self, in_dim, hgcn_dim, dropout):

        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout
    def forward(self, x, adj,G=False):

        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.1)
        if G==False:
            return x
        else:
            return x,adj
