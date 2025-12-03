import pandas as pd

from dataset_loader.load_dataset import load_dataset
from model.gcn import GCN
from explainer.gnn_explainer_wrapper import get_explanation_node, get_explanation_node_binary
from explainer.subgraphx_wrapper import SubgraphXExplainer
from explainer.subgraph_utils import get_khop_subgraph
from explainer.importance_filters import filtered_node_importance
from metrics.centralities import get_centralities
from metrics.correlations import get_correlation_centralities, get_mutual_information_centralities

# ========= Load Dataset =========
dataset, data = load_dataset("KarateClub")

# ========= Build & train model =========
model = GCN(dataset)
optimizer = None

# Quick training
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Training model...", flush=True)
for i in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# ========= Global Explanation =========
print("\nGenerating explanations...", flush=True)

results = {}
subgraph = data
centralities = get_centralities(subgraph)

def calculate_metrics(explanation, centralities, subgraph, results, prefix):
    node_imp = filtered_node_importance(explanation, subgraph)
    
    # Pearson
    corr, p_value = get_correlation_centralities(centralities, node_imp, subgraph)
    results[f"{prefix}_pearson"] = corr
    results[f"{prefix}_pearson_pvalue"] = p_value
    
    # Mutual Information
    mi = get_mutual_information_centralities(centralities, node_imp, subgraph)
    results[f"{prefix}_mi"] = mi

# 1. GNNExplainer
print("Running GNNExplainer...", flush=True)
exp_gnn = get_explanation_node(model, data)
exp_gnn_binary = get_explanation_node_binary(model, data)
calculate_metrics(exp_gnn, centralities, subgraph, results, "gnn")
calculate_metrics(exp_gnn_binary, centralities, subgraph, results, "gnn_binary")

# 2. SubgraphX
print("Running SubgraphX...", flush=True)
selected_explainer = SubgraphXExplainer( model,
                                    num_classes=dataset.num_classes,
                                    num_hops=2,
                                    explain_graph=True,
                                    rollout=5, # 20
                                    expand_atoms=5, #14
                                    device=device)
exp_subx_bin = selected_explainer.explain_graph(data, bin_weight=True)
exp_subx_score = selected_explainer.explain_graph(data, bin_weight=False)

calculate_metrics(exp_subx_bin, centralities, subgraph, results, "subx_bin")
calculate_metrics(exp_subx_score, centralities, subgraph, results, "subx_score")

# Save results
df = pd.DataFrame(results).T
print("\n📊 Correlations & MI:", flush=True)
print(df.round(4), flush=True)
df.to_csv("correlations_karateclub.csv")