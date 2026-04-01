import os.path as osp
import torch
from torch_geometric.datasets import KarateClub, Planetoid, TUDataset, ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif
from graphxai.datasets import ShapeGGen

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
    # Dataset artificial com ground truth explanation masks
    elif choice == "synthetic":
        dataset, data = synthetic_dataset()
    elif choice == "shapeggen":
        dataset = ShapeGGen(
            model_layers=2,
            num_subgraphs=10,
            subgraph_size=15,
            prob_connection=0.8,
            add_sensitive_feature=True,
            n_features=10,
            n_informative_features=3,
            seed=42,
            class_sep=1.0
        )
        dataset.num_classes = 2
        data = dataset.get_graph(use_fixed_split=True)
        # add_sensitive_feature adds an extra dimension to x
        dataset.num_features = data.x.shape[1]
        data.val_mask = data.valid_mask
        # print(data.shape.to('cpu').numpy())
        # print(data.y.to('cpu').numpy())

    return dataset, data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def synthetic_dataset():
    house = ExplainerDataset(
            graph_generator= BAGraph(num_nodes=300, num_edges=5),
            motif_generator=HouseMotif(), 
            num_motifs = 1,
            num_graphs = 50,
        )
    # cycle = ExplainerDataset(
    #         graph_generator= BAGraph(num_nodes=300, num_edges=5),
    #         motif_generator=CycleMotif(5), 
    #         num_motifs = 1,
    #         num_graphs = 50,
    #     )
    dataset = house 
    data = dataset[0]
        
    # Se não tiver características, gera algumas aleatórias
    if data.x is None:
        num_features = 10
        data.x = torch.randn((data.num_nodes, num_features))
        # Atualiza o dataset para reportar o número correto de features se necessário
        # (PyG ExplainerDataset pode não ter num_features se x for None)
            
    # Garante que tenha máscaras para o trainer
    if not hasattr(data, 'train_mask'):
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[indices[:int(0.6 * num_nodes)]] = True
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[indices[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[indices[int(0.8 * num_nodes):]] = True

    return dataset, data