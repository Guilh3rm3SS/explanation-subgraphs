import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(GCNConv(num_features, num_classes))
        else:
            self.convs.append(GCNConv(num_features, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x=None, edge_index=None, data=None):
        if data is not None:
            x = data.x
            edge_index = data.edge_index
        elif x is not None and edge_index is None and hasattr(x, 'x'):
            data = x
            x = data.x
            edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

