import os.path as osp
import torch
from torch_geometric.datasets import KarateClub, Planetoid

def load_dataset(choice="KarateClub"):
    if choice == "Cora":
        name = "Cora"
        path = osp.join(osp.dirname(osp.realpath("__file__")), "..", "data", "Planetoid")
        dataset = Planetoid(path, name)
        data = dataset[0]
    else:
        dataset = KarateClub()
        data = dataset[0]

    return dataset, data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
