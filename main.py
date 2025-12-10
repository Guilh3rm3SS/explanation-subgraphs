import pandas as pd
import os
from dataset_loader.load_dataset import load_dataset
from torch_geometric.datasets import TUDataset
from model.gcn import GCN
from explainer.gnn_explainer_wrapper import get_explainer
from explainer.subgraphx_wrapper import SubgraphXExplainer
from explainer.subgraph_utils import get_khop_subgraph
from explainer.importance_filters import filtered_node_importance
from torch_geometric.explain.metric import fidelity, characterization_score
from metrics.centralities import get_centralities
from metrics.correlations import get_correlation_centralities, get_mutual_information_centralities
import torch.nn.functional as F
import torch

# ========= Load Dataset =========
# names: PubMed, CiteSeer, KarateClub, Mutag, Cora
dataset_name = "CiteSeer"
dataset, data = load_dataset(dataset_name)

target = None
explanation_type = "phenomenon"
if explanation_type == "phenomenon":
    target = data.y

# ========= Build & train model =========
model = GCN(dataset)
optimizer = None

# Quick training

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
gnn_explainer_explanation = None
gnn_explainer_characterization_score = 0
gnn_explainer_fidelity = (0, 0)
for i in range(5):
    print(f"Running GNNExplainer {i+1}...", flush=True)
    gnn_explainer = get_explainer(model, data, algorithm="gnnexplainer", explanation_type=explanation_type, epochs=200, lr=0.1)
    it_explanation = gnn_explainer(data.x, data.edge_index, index=None, target=target)
    
    it_fidelity = fidelity(gnn_explainer, it_explanation)
    it_characterization_score = characterization_score(it_fidelity[0], it_fidelity[1])
    print("GNNExplainer fidelity:", it_fidelity)
    print("GNNExplainer characterization score:", it_characterization_score)
    if it_characterization_score > gnn_explainer_characterization_score:
        gnn_explainer_characterization_score = it_characterization_score
        gnn_explainer_fidelity = it_fidelity
        gnn_explainer_explanation = it_explanation

calculate_metrics(gnn_explainer_explanation, centralities, subgraph, results, "gnn")

# 2. PGExplainer
# print("Running PGExplainer...", flush=True)
# pg_explainer = get_explainer(model, data, algorithm="pgexplainer", epochs=200, lr=0.1)
# calculate_metrics(pg_explainer, centralities, subgraph, results, "pg")

# 3. GraphMask
# print("Running GraphMask...", flush=True)
# graphmask_explainer = get_explainer(model, data, algorithm="graphmask", explanation_type=explanation_type, epochs=200, lr=0.1)
# calculate_metrics(graphmask_explainer, centralities, subgraph, results, "graphmask")

# 4. Captum
# print("Running Captum...", flush=True)
# captum_explainer = get_explainer(model, data, algorithm="captum", epochs=200, lr=0.1)
# calculate_metrics(captum_explainer, centralities, subgraph, results, "captum")

# exp_gnn_binary = get_explainer_binary(model, data)
# calculate_metrics(exp_gnn_binary, centralities, subgraph, results, "gnn_binary")

# 2. SubgraphX
# print("Running SubgraphX...", flush=True)
# selected_explainer = SubgraphXExplainer( model,
#                                     num_classes=dataset.num_classes,
#                                     num_hops=2,
#                                     explain_graph=True,
#                                     rollout=5, # 20
#                                     expand_atoms=5, #14
#                                     device=device)
# exp_subx_bin = selected_explainer.explain_graph(data, bin_weight=True)
# exp_subx_score = selected_explainer.explain_graph(data, bin_weight=False)

# calculate_metrics(exp_subx_bin, centralities, subgraph, results, "subx_bin")
# calculate_metrics(exp_subx_score, centralities, subgraph, results, "subx_score")


# PGexplainer

# Save results
df = pd.DataFrame(results).T
print("\n📊 Correlations & MI:", flush=True)
print(df.round(4), flush=True)

results_base_folder = f"results/{dataset_name}"
os.makedirs(results_base_folder, exist_ok=True)

# Generate unique run folder
run_id = 1
while True:
    run_folder_name = f"run_{run_id:03d}"
    run_folder_path = os.path.join(results_base_folder, run_folder_name)
    if not os.path.exists(run_folder_path):
        os.makedirs(run_folder_path)
        break
    run_id += 1

filename = f"correlations.csv"
file_path = os.path.join(run_folder_path, filename)

# Save
df.to_csv(file_path, index=True)
print("Salvo em:", file_path)

with open(f"{run_folder_path}/fidelity.txt", "w") as f:
    print(f"positive fidelity: {gnn_explainer_fidelity[0]}", file=f)
    print(f"negative fidelity: {gnn_explainer_fidelity[1]}", file=f)
    print(f"characterization score: {gnn_explainer_characterization_score}", file=f)
