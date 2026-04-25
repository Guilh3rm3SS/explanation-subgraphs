import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, Planetoid, TUDataset, ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif
from graphxai.datasets import ShapeGGen, AlkaneCarbonyl, Benzene

dataset = Benzene()
for i in range (len(dataset)):
    graph, exps = dataset[i]
    print(dataset[0])
    print(graph)
    print(graph.y)
    print(len(exps))
    print(exps[0].node_imp)
    print(i)

# graph, exps = dataset[1124]
# print(graph)
# print(graph.y)
# print(len(exps))
# for i in range(7):
#     print(exps[i].node_imp)
#     print(i)
#     exps[i].visualize_graph(show=True)