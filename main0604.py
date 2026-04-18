import pandas as pd
import os
from dataset_loader.load_dataset import load_dataset, filter_explanation_by_mask
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

node_mask_type = "object"  # None, attributes, object
edge_mask_type = "object"
target = None
index = 13
explanation_type = "phenomenon"
if explanation_type == "phenomenon":
    target = data.y
# print(data)



# Model type: gcn or gatconv
model_type = "gcn"

# Treinar ou carregar o modelo, caso já exista
model, model_params = None, None
# model, model_params = get_model_checkpoint(model_type, dataset_name, data.num_features, dataset.num_classes) if dataset_name != "synthetic" else (None, None)


if model is None:
    print("Checkpoint não encontrado. Treinando modelo com melhores parâmetros...", flush=True)
    model, model_params = optimize_hyperparameters(data, dataset, model_type)
    save_model_checkpoint(model, model_type, model_params, dataset_name) if dataset_name != "synthetic" else None
else:
    print(f"Checkpoint encontrado para {model_type} no dataset {dataset_name}.", flush=True)






# ========= Visualization Helper =========
def visualize_subgraph(subgraph_data, target_index, node_importance, title, filename, threshold=0.1, edge_importance=None):
    # Print nodes and edges
    print(f"\n--- {title} ---")
    if hasattr(subgraph_data, 'node_idx_original'):
        nodes = subgraph_data.node_idx_original.tolist()
    else:
        nodes = list(range(subgraph_data.num_nodes))
    # print(f"Nodes ({len(nodes)}): {nodes}")
    
    # Get edges
    edge_index = subgraph_data.edge_index
    edges = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edges.append((u, v))
    # print(f"Edges ({len(edges)}): {edges}")

    # Plot
    # We use the original indices as node labels in NetworkX
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=43)
    
    edge_colors = 'black'
    edge_labels = {}
    if edge_importance is not None:
        edge_colors = []
        for u, v in G.edges():
            val1 = edge_importance.get((u, v), 0.0)
            val2 = edge_importance.get((v, u), 0.0)
            val = max(val1, val2)
            
            if val > threshold:
                edge_colors.append('red')
            else:
                edge_colors.append('black')
            edge_labels[(u, v)] = f"{val:.2f}"
            
    node_colors = []
    custom_labels = {}
    for i, node in enumerate(nodes):
        val = node_importance[i] if len(node_importance) == len(nodes) else node_importance[node]
        
        # Obter a label original (y) do nó
        y_label = ""
        if hasattr(subgraph_data, 'y') and subgraph_data.y is not None:
            if len(subgraph_data.y) == len(nodes):
                y_label = f"\ny={subgraph_data.y[i].item()}"
            elif len(subgraph_data.y) > node:
                y_label = f"\ny={subgraph_data.y[node].item()}"
        
        # Define a cor do nó
        if node == target_index:
            if val > threshold:
                node_colors.append('orange')
            else:
                node_colors.append('yellow')
        else:
            if val > threshold:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
                
        # Adiciona a importância e a classe (y) no rótulo
        custom_labels[node] = f"{node}{y_label}\n{val:.2f}"
            
    plt.figure(figsize=(9, 7))
    nx.draw(G, pos, labels=custom_labels, node_color=node_colors, edge_color=edge_colors, node_size=1200, font_size=8, font_weight='bold')
    
    if edge_importance is not None:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

# ========= Global Explanation =========
print("\nGerando explicações...", flush=True)

results = {}
test_indices = data.test_mask.nonzero(as_tuple=True)[0]
data_test = data
# centralities = get_centralities(data_test)

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
shape_vector = (data_test.shape > 0).int() if getattr(data_test, "shape", None) is not None else None

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

    it_explanation = gnn_explainer(data_test.x, data_test.edge_index, index=index, target=target)
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

prediction_vector_np = None
shape_vector_np = shape_vector.detach().cpu().numpy() if shape_vector is not None else None
visual_graph = data_test

if getattr(gnn_explainer_explanation, 'node_mask', None) is not None:
    node_mask = gnn_explainer_explanation.node_mask
    if node_mask.dim() > 1:
        node_mask = node_mask.mean(dim=-1)
    prediction_vector_np = node_mask.detach().cpu().float().numpy()
    if prediction_vector_np.max() > 0:
        prediction_vector_np = prediction_vector_np / prediction_vector_np.max()

edge_importance_map = None
edge_shape_map = None
if getattr(gnn_explainer_explanation, 'edge_mask', None) is not None:
    edge_mask_np = gnn_explainer_explanation.edge_mask.detach().cpu().numpy()
    if edge_mask_np.max() > 0:
        edge_mask_np = edge_mask_np / edge_mask_np.max()

    edge_importance_map = {}
    edge_shape_map = {}
    for k in range(data_test.edge_index.shape[1]):
        u = data_test.edge_index[0, k].item()
        v = data_test.edge_index[1, k].item()
        edge_importance_map[(u, v)] = edge_mask_np[k]
        if shape_vector_np is not None:
            # An edge belongs to a shape if both incident nodes are in the shape vector
            if shape_vector_np[u] > 0 and shape_vector_np[v] > 0:
                edge_shape_map[(u, v)] = 1.0
            else:
                edge_shape_map[(u, v)] = 0.0

if index is not None:
    gt_subgraph = get_khop_subgraph(data_test, index, num_hops=model_params['layers'], relabel_nodes=False)
    
    if prediction_vector_np is not None:
        prediction_vector_np = prediction_vector_np[gt_subgraph.node_idx_original]
    
    if shape_vector_np is not None:
        shape_vector_np = shape_vector_np[gt_subgraph.node_idx_original]
        
    visual_graph = gt_subgraph

metrics = {}
if prediction_vector_np is not None and shape_vector_np is not None:
    gnn_explainer_gt_metrics_nodes = groundtruth_metrics(
        torch.tensor(prediction_vector_np), 
        torch.tensor(shape_vector_np)
    )
    metrics["nodes"] = gnn_explainer_gt_metrics_nodes

if edge_importance_map is not None and edge_shape_map is not None:
    edge_pred = []
    edge_true = []
    # If a subgraph is extracted, evaluate only on its edges
    for k in range(visual_graph.edge_index.shape[1]):
        u = visual_graph.edge_index[0, k].item()
        v = visual_graph.edge_index[1, k].item()
        edge_pred.append(edge_importance_map.get((u, v), 0.0))
        edge_true.append(edge_shape_map.get((u, v), 0.0))
        
    gnn_explainer_gt_metrics_edges = groundtruth_metrics(
        torch.tensor(edge_pred),
        torch.tensor(edge_true)
    )
    metrics["edges"] = gnn_explainer_gt_metrics_edges

if node_mask_type is not None:
    print("\nMétricas de Ground Truth (nodes):", metrics["nodes"])
if edge_mask_type is not None:
    print("\nMétricas de Ground Truth (edges):", metrics["edges"])

# Visualization
num_nodes_vis = len(visual_graph.node_idx_original) if hasattr(visual_graph, 'node_idx_original') else visual_graph.num_nodes
dummy_node_importance_gt = shape_vector_np if shape_vector_np is not None else [0.0] * num_nodes_vis
dummy_node_importance_pred = prediction_vector_np if prediction_vector_np is not None else [0.0] * num_nodes_vis

visualize_subgraph(visual_graph, index, dummy_node_importance_gt, "Ground Truth Subgraph", "gt_subgraph.png", threshold=0.0, edge_importance=edge_shape_map)
visualize_subgraph(visual_graph, index, dummy_node_importance_pred, "Explanation Subgraph", "expl_subgraph.png", threshold=0.1, edge_importance=edge_importance_map)

    # print("Quantidade de nós importantes preditos no subgrafo:", prediction_vector_sub.sum())
    # print("Quantidade de nós importantes no subgrafo real:", shape_vector_sub.sum())

    # print("proporção de nós importantes preditos no subgrafo:", prediction_vector_sub.sum() / len(shape_vector_sub))
    # print("proporção de nós importantes no subgrafo real:", shape_vector_sub.sum() / len(shape_vector_sub))



    
















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
