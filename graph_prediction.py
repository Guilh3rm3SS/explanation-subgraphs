import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric import groundtruth_metrics

from dataset_loader.load_dataset import load_molecule_datasets
from model.gcn import GraphGCN
from model.trainer import optimize_hyperparameters, get_model_checkpoint, save_model_checkpoint, evaluate_model
from metrics.fidelity import get_fidelity_metrics

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']

def visualize_graph(data, node_importance=None, edge_importance=None, title="Graph", threshold=0.5):
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    pos = nx.spring_layout(G, seed=42)
    
    # Convert node_importance to numpy if it's a tensor
    if torch.is_tensor(node_importance):
        node_importance = node_importance.detach().cpu().numpy()
            
    node_colors = []
    labels = {}
    
    # Map features to symbols
    x = data.x.cpu().numpy()
    
    for i in range(data.num_nodes):
        # Get atom symbol
        atom_idx = x[i].argmax()
        symbol = ATOM_TYPES[atom_idx] if atom_idx < len(ATOM_TYPES) else str(i)
        
        val = node_importance[i] if node_importance is not None else 0
        if node_importance is not None and val > threshold:
            node_colors.append('red')
            labels[i] = f"{symbol}\n{val:.2f}"
        else:
            node_colors.append('lightblue')
            labels[i] = symbol
            
    edge_colors = []
    if edge_importance is not None:
        if torch.is_tensor(edge_importance):
            edge_importance = edge_importance.detach().cpu().numpy()
            
        for u, v in G.edges():
            # Find edge index
            idx = -1
            for k in range(data.edge_index.shape[1]):
                if (data.edge_index[0, k] == u and data.edge_index[1, k] == v) or \
                   (data.edge_index[1, k] == u and data.edge_index[0, k] == v):
                    idx = k
                    break
            if idx != -1 and edge_importance[idx] > threshold:
                edge_colors.append('red')
            else:
                edge_colors.append('black')
    else:
        edge_colors = 'black'
        
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, labels=labels, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=800, font_size=8)
    plt.title(title)

def main():
    dataset_name = "Benzene"
    print(f"Loading dataset: {dataset_name}...")
    data_dict = load_molecule_datasets(choice=dataset_name, split=(0.7, 0.2, 0.1))
    
    print(f"Number of training graphs: {len(data_dict['train'][0])}")
    print(f"Number of validation graphs: {len(data_dict['val'][0])}")
    print(f"Number of test graphs: {len(data_dict['test'][0])}")
    print(f"Num features: {data_dict['num_features']}")
    print(f"Num classes: {data_dict['num_classes']}")
    
    model_type = "gcn_graph"
    # Load or train model
    model, model_params = get_model_checkpoint(model_type, dataset_name, data_dict['num_features'], data_dict['num_classes'])
    
    if model is None:
        print("Checkpoint not found. Optimizing hyperparameters...")
        model, model_params = optimize_hyperparameters(data_dict, model_type=model_type)
        save_model_checkpoint(model, model_type, model_params, dataset_name)
    else:
        print(f"Loaded checkpoint for {model_type} on {dataset_name}")
        
    print("\nEvaluating model on test set...")
    f1, recall, precision = evaluate_model(model, data_dict)
    print(f"Test Set Metrics - F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
        
    print("\nSetting up explainer...")
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=500),
        explanation_type='phenomenon',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    )
    
    test_graphs, test_masks = data_dict['test']
    num_samples = 3
    
    for idx in range(num_samples+3):
        sample_data = test_graphs[idx]
        sample_label = sample_data.y
        sample_gt = test_masks[idx]
        
        print(f"\n--- Graph {idx} (Label: {sample_label.item()}) ---")
        it_explanation = explainer(sample_data.x, sample_data.edge_index, target=sample_label)
        
        # Fidelity Metrics
        metrics_fid = get_fidelity_metrics(it_explanation, explainer)
        print("Fidelity:", metrics_fid)
        
        # Node importance
        node_mask = it_explanation.node_mask
        if node_mask.dim() > 1: node_mask = node_mask.mean(dim=-1)
        pred_nodes = node_mask.detach().cpu()
        if pred_nodes.max() > 0: pred_nodes = pred_nodes / pred_nodes.max()
        
        # Ground Truth
        gt_nodes = sample_gt.detach().cpu()
        metrics_gt = groundtruth_metrics(pred_nodes, gt_nodes)
        print("Ground Truth (Nós):", metrics_gt)
        
        # Visualization
        visualize_graph(sample_data, node_importance=pred_nodes, title=f"Explanation {idx}", threshold=0.2)
        plt.savefig(f"explanation_{idx}.png")
        plt.close()
        
        visualize_graph(sample_data, node_importance=gt_nodes, title=f"Ground Truth {idx}", threshold=0.5)
        plt.savefig(f"gt_{idx}.png")
        plt.close()
        print(f"Saved plots for graph {idx}")
        
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
