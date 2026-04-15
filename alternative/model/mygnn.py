import torch
from torch_geometric.nn import GINConv

class MyGNN(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, classes=2):
        super(MyGNN, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, classes)
        self.gin2 = GINConv(self.mlp_gin2)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
        return x
