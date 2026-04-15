import pandas as pd
import os
from dataset_loader.load_dataset import load_dataset, filter_explanation_by_mask
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.explain.metric import groundtruth_metrics
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
# names: PubMed, CiteSeer, KarateClub, Mutag, Cora, synthetic, shapeggen
dataset_name = "shapeggen"
dataset, data = load_dataset(dataset_name)
print(data)

node_mask_type = "attributes"
target = None
explanation_type = "model"
if explanation_type == "phenomenon":
    target = data.y
index = 0

# Model type: gcn or gatconv
model_type = "gcn"

# ========= Build & train model =========
# model, model_params = get_model_checkpoint(model_type, dataset_name, data.num_features, dataset.num_classes) if dataset_name != "synthetic" else (None, None)
model, model_params = None, None


if model is None:
    print("Checkpoint não encontrado. Treinando modelo com melhores parâmetros...", flush=True)
    model, model_params = optimize_hyperparameters(data, dataset, model_type)
    save_model_checkpoint(model, model_type, model_params, dataset_name) if dataset_name != "synthetic" else None
else:
    print(f"Checkpoint encontrado para {model_type} no dataset {dataset_name}.", flush=True)
# model, loss = train_one_model(    data, 
#                             num_features=dataset.num_features, 
#                             num_classes=dataset.num_classes, 
#                             hidden_channels=32, 
#                             num_layers=3, 
#                             dropout=0.25, 
#                             epochs=200, 
#                             lr=0.1, 
#                             patience=10)

# ========= Global Explanation =========
print("\nGerando explicações...", flush=True)

results = {}
subgraph = filter_explanation_by_mask(data, data.test_mask)
print(subgraph)
# centralities = get_centralities(subgraph)

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
gnn_explainer_fidelity_metrics = None
gnn_explainer_characterization_score = -1
gnn_explainer_epochs = 500
gnn_explainer_lr = 0.1
for i in range(1):
    print(f"Rodando GNNExplainer {i+1}...", flush=True)
    gnn_explainer = get_explainer(  model,
                                    data,
                                    algorithm="gnnexplainer", 
                                    explanation_type=explanation_type, 
                                    epochs=gnn_explainer_epochs, 
                                    lr=gnn_explainer_lr, 
                                    node_mask_type=node_mask_type, 
                                    num_layers=model_params['layers'])

    it_explanation = gnn_explainer(subgraph.x, subgraph.edge_index, index=None, target=target)
    
    metrics = get_fidelity_metrics(it_explanation, gnn_explainer)
    
    print("Métricas do GNNExplainer:", metrics)
    
    if metrics["characterization_score"] > gnn_explainer_characterization_score:
        gnn_explainer_characterization_score = metrics["characterization_score"]
        gnn_explainer_fidelity_metrics = metrics
        gnn_explainer_explanation = it_explanation

# ========= Ground Truth Comparison =========
if dataset_name == "shapeggen" and gnn_explainer_explanation is not None:
    print("Calculando métricas de Ground Truth...", flush=True)
    
    # Se a máscara for de atributos (node_mask_type="attributes"), 
    # precisamos reduzir para nível de nó para comparar com o Ground Truth
    explanation_node_mask = gnn_explainer_explanation.node_mask
    gt_node_mask = subgraph.shape
    if explanation_node_mask is not None and explanation_node_mask.dim() > 1:
        explanation_node_mask = explanation_node_mask.mean(dim=-1)
    
    gt_metrics = groundtruth_metrics(explanation_node_mask, gt_node_mask)
    print("Métricas de Ground Truth:", gt_metrics)
    
    # groundtruth_metrics retorna (accuracy, recall, precision, f1, auc)
    acc, recall, prec, f1, auc, jaccard = gt_metrics
    results["gnn_gt_accuracy"] = acc
    results["gnn_gt_recall"] = recall
    results["gnn_gt_precision"] = prec
    results["gnn_gt_f1"] = f1
    results["gnn_gt_auc"] = auc
    results["gnn_gt_jaccard"] = jaccard
else:
    gt_metrics = None


# calculate_metrics(gnn_explainer_explanation, centralities, subgraph, results, "gnn")

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
# df = pd.DataFrame(results).T
# print("\n📊 Correlações & MI:", flush=True)
# print(df.round(4), flush=True)

# results_base_folder = f"results/{dataset_name}"
# os.makedirs(results_base_folder, exist_ok=True)

# # Generate unique run folder
# run_id = 1
# while True:
#     run_folder_name = f"run_{run_id:03d}"
#     run_folder_path = os.path.join(results_base_folder, run_folder_name)
#     if not os.path.exists(run_folder_path):
#         os.makedirs(run_folder_path)
#         break
#     run_id += 1

# filename = f"correlations.csv"
# file_path = os.path.join(run_folder_path, filename)

# # Save
# df.to_csv(file_path, index=True)
# print("Salvo em:", file_path)

# with open(f"{run_folder_path}/fidelity.txt", "w") as f:
#     if gnn_explainer_fidelity_metrics:
#         for key, value in gnn_explainer_fidelity_metrics.items():
#             print(f"{key}: {value}", file=f)
#     # if gt_metrics:
#     #     print("\n--- Ground Truth Metrics ---", file=f)
#     #     for key, value in gt_metrics.items():
#     #         print(f"{key}: {value}", file=f)
# with open(f"{run_folder_path}/parameters.txt", "w") as f:
#     print(f"epochs: {gnn_explainer_epochs}", file=f)
#     print(f"lr: {gnn_explainer_lr}", file=f)
#     print(f"explanation_type: {explanation_type}", file=f)
#     print(f"model_type: {model_type}", file=f)
#     print(f"model_params: {model_params}", file=f)
