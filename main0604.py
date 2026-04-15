import pandas as pd
import os
from dataset_loader.load_dataset import load_dataset
from torch_geometric.datasets import TUDataset
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
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# ========= Load Dataset =========

# names: PubMed, CiteSeer, KarateClub, Mutag, Cora, synthetic, shapeggen
dataset_name = "shapeggen"
dataset, data = load_dataset(dataset_name)

node_mask_type = "attributes"  # None, attributes, object
edge_mask_type = None
target = None
index = 0
explanation_type = "phenomenon"
if explanation_type == "phenomenon":
    target = data.y
# print(data)



# Model type: gcn or gatconv
model_type = "gcn"

# Treinar ou carregar o modelo, caso já exista
# model, model_params = get_model_checkpoint(model_type, dataset_name, data.num_features, dataset.num_classes) if dataset_name != "synthetic" else (None, None)

model, model_params = None, None

if model is None:
    print("Checkpoint não encontrado. Treinando modelo com melhores parâmetros...", flush=True)
    model, model_params = optimize_hyperparameters(data, dataset, model_type)
    save_model_checkpoint(model, model_type, model_params, dataset_name) if dataset_name != "synthetic" else None
else:
    print(f"Checkpoint encontrado para {model_type} no dataset {dataset_name}.", flush=True)






# ========= Visualization Helper =========
def visualize_subgraph(subgraph_data, target_index, node_importance, title, filename, threshold=0.1):
    # Print nodes and edges
    print(f"\n--- {title} ---")
    nodes = subgraph_data.node_idx_original.tolist()
    print(f"Nodes ({len(nodes)}): {nodes}")
    
    # Get edges
    edge_index = subgraph_data.edge_index
    edges = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edges.append((u, v))
    print(f"Edges ({len(edges)}): {edges}")

    # Plot
    # We use the original indices as node labels in NetworkX
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = []
    for node in nodes:
        if node == target_index:
            node_colors.append('yellow')
        elif node_importance[node] > threshold:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
            
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_weight='bold')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

# ========= Global Explanation =========
print("\nGerando explicações...", flush=True)

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
gnn_explainer_fidelity_metrics = None
gnn_explainer_gt_metrics = None
gnn_explainer_characterization_score = -1
gnn_explainer_epochs = 500
gnn_explainer_lr = 0.01
shape_vector = (data.shape > 0).int() if getattr(data, "shape", None) is not None else None
for i in range(1):
    print(f"Rodando GNNExplainer {i+1}...", flush=True)
    gnn_explainer = get_explainer(  model,
                                    data,
                                    algorithm="gnnexplainer", 
                                    explanation_type=explanation_type, 
                                    epochs=gnn_explainer_epochs, 
                                    lr=gnn_explainer_lr, 
                                    node_mask_type=node_mask_type, 
                                    edge_mask_type=edge_mask_type,
                                    num_layers=model_params['layers'])

    it_explanation = gnn_explainer(data.x, data.edge_index, index=index, target=target)
    metrics = get_fidelity_metrics(it_explanation, gnn_explainer)
    
    # print("Métricas do GNNExplainer:", metrics)
    
    if metrics["characterization_score"] > gnn_explainer_characterization_score:
        gnn_explainer_characterization_score = metrics["characterization_score"]
        gnn_explainer_fidelity_metrics = metrics
        gnn_explainer_explanation = it_explanation
        # gnn_explainer_gt_metrics = groundtruth_metrics(it_explanation.node_mask, shape_vector)
        # print("Métricas de Ground Truth:", gnn_explainer_gt_metrics)

print("Melhor GNNExplainer:", gnn_explainer_fidelity_metrics)
# print(gnn_explainer_explanation.node_mask.flatten())
# print("Melhor GNNExplainer GT:", gnn_explainer_gt_metrics)

prediction_vector = gnn_explainer_explanation.node_mask.flatten().detach().cpu()
prediction_vector = (prediction_vector > 0).float().numpy()
shape_vector_np = shape_vector.detach().cpu().numpy()
# print(prediction_vector)
# print(shape_vector_np)

if index is not None:
    gt_subgraph = get_khop_subgraph(data, index, num_hops=model_params['layers'], relabel_nodes=False)
    # print(gt_subgraph.node_idx_original)
    #subgraph = gnn_explainer_explanation.get_explanation_subgraph()

    prediction_vector_sub = prediction_vector[gt_subgraph.node_idx_original]
    shape_vector_sub = shape_vector_np[gt_subgraph.node_idx_original]


    print(prediction_vector_sub)
    print(shape_vector_sub)

    gnn_explainer_gt_metrics = groundtruth_metrics(
        torch.tensor(prediction_vector_sub), 
        torch.tensor(shape_vector_sub)
    )
    print(gnn_explainer_gt_metrics)

    # Visualization
    visualize_subgraph(gt_subgraph, index, shape_vector_np, "Ground Truth Subgraph", "gt_subgraph.png", threshold=0.0)
    visualize_subgraph(gt_subgraph, index, prediction_vector, "Explanation Subgraph", "expl_subgraph.png", threshold=0.1)

    print("Quantidade de nós importantes preditos no subgrafo:", prediction_vector_sub.sum())
    print("Quantidade de nós importantes no subgrafo real:", shape_vector_sub.sum())

    print("proporção de nós importantes preditos no subgrafo:", prediction_vector_sub.sum() / len(shape_vector_sub))
    print("proporção de nós importantes no subgrafo real:", shape_vector_sub.sum() / len(shape_vector_sub))



    
















# print(prediction_vector)
# print(shape_vector)



    

# # ========= Ground Truth Comparison =========
# if dataset_name == "synthetic" and gnn_explainer_explanation is not None:
#     print("Calculando métricas de Ground Truth...", flush=True)
    
#     # Se a máscara for de atributos (node_mask_type="attributes"), 
#     # precisamos reduzir para nível de nó para comparar com o Ground Truth
#     node_mask = gnn_explainer_explanation.node_mask
#     if node_mask is not None and node_mask.dim() > 1:
#         node_mask = node_mask.mean(dim=-1)
    
    
#     # groundtruth_metrics retorna (accuracy, recall, precision, f1, auc)
#     acc, recall, prec, f1, auc = gt_metrics
#     results["gnn_gt_accuracy"] = acc
#     results["gnn_gt_recall"] = recall
#     results["gnn_gt_precision"] = prec
#     results["gnn_gt_f1"] = f1
#     results["gnn_gt_auc"] = auc
# else:
#     gt_metrics = None

# calculate_metrics(gnn_explainer_explanation, centralities, subgraph, results, "gnn")


# # Save results
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
