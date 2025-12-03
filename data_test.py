import pandas as pd

from dataset_loader.load_dataset import load_dataset
from model.gcn import GCN
from explainer.gnn_explainer_wrapper import get_explanation_node
from explainer.subgraphx_wrapper import SubgraphXExplainer
from explainer.subgraph_utils import get_khop_subgraph
from explainer.importance_filters import filtered_node_importance
from metrics.centralities import get_centralities
from metrics.correlations import get_correlation_centralities

# ========= Load Dataset =========
dataset, data = load_dataset("KarateClub")

print(dataset.num_classes)
print(data)