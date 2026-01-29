import pandas as pd
import os
from dataset_loader.load_dataset import load_dataset
from torch_geometric.datasets import TUDataset
from model.gcn import GCN
from model.trainer import optimize_hyperparameters, train_one_model, get_model_checkpoint, save_model_checkpoint
from explainer.gnn_explainer_wrapper import get_explainer
from explainer.subgraphx_wrapper import SubgraphXExplainer
from explainer.subgraph_utils import get_khop_subgraph
from explainer.importance_filters import filtered_node_importance
from metrics.fidelity import get_fidelity_metrics
from metrics.centralities import get_centralities
from metrics.correlations import get_correlation_centralities, get_mutual_information_centralities
import torch.nn.functional as F
import torch

# ========= Load Dataset =========
# names: PubMed, CiteSeer, KarateClub, Mutag, Cora
dataset_name = "synthetic"
dataset, data = load_dataset(dataset_name)

print(data.keys())
print(dataset[1])

# ['val_mask', 'x', 'y', 'test_mask', 'edge_mask', 'node_mask', 'edge_index', 'train_mask']
# Explanation(edge_index=[2, 2850], y=[305], edge_mask=[2850], node_mask=[305])