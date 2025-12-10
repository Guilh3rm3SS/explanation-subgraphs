import os.path as osp
import torch
from torch_geometric.datasets import KarateClub, Planetoid, TUDataset

def load_dataset(choice="KarateClub"):
    if choice == "Cora":
        name = "Cora"
        path = osp.join(osp.dirname(osp.realpath("__file__")), "..", "data", "Planetoid")
        dataset = Planetoid(path, name)
        data = dataset[0]
    elif choice == "KarateClub":
        dataset = KarateClub()
        data = dataset[0]
    elif choice == "MUTAG":
        dataset = TUDataset(root='data/TUDataset', name='MUTAG')
        data = dataset[0]
    elif choice == "CiteSeer":
        name = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath("__file__")), "..", "data", "Planetoid")
        dataset = Planetoid(path, name)
        data = dataset[0]
    elif choice == "PubMed":
        name = "PubMed"
        path = osp.join(osp.dirname(osp.realpath("__file__")), "..", "data", "Planetoid")
        dataset = Planetoid(path, name)
        data = dataset[0]

    return dataset, data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
