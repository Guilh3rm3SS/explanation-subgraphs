import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, Planetoid, TUDataset, ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif
from graphxai.datasets import ShapeGGen
from .generators import SyntheticGraphGenerator

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
            shape="house",
            subgraph_size=10,
            prob_connection=1,
            add_sensitive_feature=True,
            n_features=10,
            n_informative_features=3,
            seed=42,
            class_sep=0.9
        )
        dataset.num_classes = 2
        data = dataset.get_graph(use_fixed_split=True)
        # add_sensitive_feature adds an extra dimension to x
        dataset.num_features = data.x.shape[1]
        data.val_mask = data.valid_mask
        # print(data.shape.to('cpu').numpy())
        # print(data.y.to('cpu').numpy())
        print(dataset)
        print("Quantidade de classes: 0: ", (data.y == 0).sum().item(), "1: ", (data.y == 1).sum().item())
        print("Quantidade de shapes: ", (data.shape == 0).sum().item(), (data.shape > 0).sum().item())


    # elif choice == "Paulo":
    #     generator = SyntheticGraphGenerator(num_nodes=200, num_houses=10) # Menor para visualizar fácil
    #     data = generator.generate()

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


def filter_explanation_by_mask(exp, mask):
    """
    Filtra um grafo/explanation pelo mask de nós e retorna um Data limpo.
    """

    mask = mask.bool()
    num_nodes = mask.size(0)

    # -------------------------
    # 1. Mapear nós antigos -> novos índices
    # -------------------------
    old_to_new = -torch.ones(num_nodes, dtype=torch.long)
    old_to_new[mask] = torch.arange(mask.sum())

    # -------------------------
    # 2. Filtrar nós
    # -------------------------
    x = exp.x[mask] if hasattr(exp, "x") else None
    y = exp.y[mask] if hasattr(exp, "y") else None

    # -------------------------
    # 3. Filtrar arestas (ambos nós precisam estar no mask)
    # -------------------------
    src, dst = exp.edge_index
    edge_keep = mask[src] & mask[dst]

    edge_index = exp.edge_index[:, edge_keep]
    edge_index = old_to_new[edge_index]

    # -------------------------
    # 4. Criar Data final
    # -------------------------
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    # -------------------------
    # 5. Copiar masks se existirem
    # -------------------------
    for attr in ["train_mask", "val_mask", "valid_mask", "test_mask", "shape"]:
        if hasattr(exp, attr):
            data[attr] = getattr(exp, attr)[mask]

    return data
