import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from graphxai.datasets import ShapeGGen
from graphxai.gnn_models.node_classification import train, test
from graphxai.explainers import PGExplainer, IntegratedGradExplainer, GradCAM
from graphxai.metrics import graph_exp_acc
import os
from model.trainer import optimize_hyperparameters
from metrics.centralities import get_centralities
from metrics.correlations import get_correlation_centralities, get_mutual_information_centralities

# Configuration
DATASET_IMAGE_PATH = '/home/guilhermess/explanation-subgraphs/shapeggen/shapeggen_dataset.png'
EXPLANATION_IMAGE_PATH = '/home/guilhermess/explanation-subgraphs/shapeggen/shapeggen_explanation.png'
EPOCHS = 300
LR = 0.001
WEIGHT_DECAY = 0.001
HIDDEN_CHANNELS = 32
PG_EPOCHS = 100
PG_LR = 0.003

class MyGNN(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, classes=2):
        super(MyGNN, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, classes)
        self.gin2 = GINConv(self.mlp_gin2)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
        return x

def get_dataset(seed=42):
    """Initializes and returns the ShapeGGen dataset."""
    print(f"Generating dataset with seed={seed}...")
    dataset = ShapeGGen(
            model_layers=2,
            num_subgraphs=10,
            subgraph_size=15,
            prob_connection=1,
            add_sensitive_feature=True,
            n_features=10,
            n_informative_features=3,
            seed=seed,
            class_sep=1.0
        )
    return dataset

def save_dataset_visualization(dataset, path):
    """Visualizes the dataset and saves the image."""
    print(f"Saving dataset visualization to {path}...")
    plt.figure(figsize=(8, 8))
    dataset.visualize(show=False)
    plt.savefig(path)
    plt.close()

def train_gnn(dataset, data, epochs=EPOCHS):
    """Trains the GNN model."""
    print("Training GNN model...")
    model = MyGNN(dataset.n_features, HIDDEN_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        train(model, optimizer, criterion, data)
    
    return model

def evaluate_model(model, data):
    """Evaluates the trained model."""
    print("Evaluating model...")
    f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes=2, get_auc=True)
    print('Test F1 score: {:.4f}'.format(f1))
    print('Test AUROC: {:.4f}'.format(auroc))
    return f1, auroc

def get_global_explanation(explainer, data, method='sum'):
    # Aggregate local explanations for all nodes in the validation/test set (or all nodes) to form a global map
    # Using all nodes for a complete global picture
    display_method = method # sum or max
    print(f"Generating global explanation using aggregate method: {display_method}")
    num_nodes = data.num_nodes
    global_imp = torch.zeros(num_nodes)
    
    # Iterate over all nodes to get their local importance
    for node_idx in range(num_nodes):
        try:
            # Get explanation for the predicted class
            exp = explainer.get_explanation_node(
                node_idx=node_idx, 
                x=data.x, 
                edge_index=data.edge_index,
                label=data.y[node_idx].item(),
                y=data.y # Required by GradCAM
            )
            
            if exp.node_imp is not None:
                # If node_imp is local indices, we need to map back to global
                if hasattr(exp, 'enc_subgraph') and exp.enc_subgraph is not None:
                    # Map local indices to global
                    local_imp = exp.node_imp
                    global_indices = exp.enc_subgraph.nodes
                    
                    if local_imp.shape == global_indices.shape:
                         global_imp[global_indices] += local_imp.cpu()
                    else:
                         pass
                else:
                    # Assumes global mask if no enc_subgraph
                    if exp.node_imp.shape[0] == num_nodes:
                        global_imp += exp.node_imp.cpu()
        except Exception:
            # Skip errors silently for production run
            pass

    # Normalize
    if global_imp.max() > global_imp.min():
         global_imp = (global_imp - global_imp.min()) / (global_imp.max() - global_imp.min())
    else:
         print("Warning: Uniform importance scores.")
         
    return global_imp

def evaluate_global_explanation(global_imp, data, name="Explainer"):
    # Ground Truth: nodes with shape > 0
    gt_mask = (data.shape > 0).cpu().numpy().astype(int)
    pred_mask = (global_imp > 0.5).cpu().numpy().astype(int) # Threshold at 0.5
    pred_scores = global_imp.cpu().numpy()
    
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    
    f1 = f1_score(gt_mask, pred_mask)
    auc = roc_auc_score(gt_mask, pred_scores)
    prec = precision_score(gt_mask, pred_mask)
    rec = recall_score(gt_mask, pred_mask)
    
    print(f"--- Global Explanation Metrics ({name}) ---")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    
    return {'F1': f1, 'AUC': auc, 'Precision': prec, 'Recall': rec}

def run_global_explanations(model, dataset, data, save_path):
    print("Running GLOBAL explainers...")
    
    # Calculate Centralities (Degree, Betweenness, etc.)
    print("Calculating Graph Centralities...")
    centralities = get_centralities(data)
    
    # PGExplainer
    # We want the layer PRODUCING embeddings. Usually the one before the classifier.
    # In GCN, this is the second to last layer (index -2).
    # If len(model.convs) > 1:
    if len(model.convs) > 1:
        target_layer_idx = len(model.convs) - 2
    else:
        target_layer_idx = 0

    target_layer = model.convs[target_layer_idx]
    model.explanation_emb_layer = target_layer
    
    # Debug print
    print(f"Target Layer: {target_layer}, Out Channels: {target_layer.out_channels}")
    
    # PGExplainer takes embedding as input (output of target_layer).
    # It concatenates embeddings of edge endpoints (u, v), so input dim is 2 * embedding_dim.
    pgex = PGExplainer(model, emb_layer_name='explanation_emb_layer', in_channels=2 * target_layer.out_channels, max_epochs=PG_EPOCHS, lr= PG_LR)
    pgex.train_explanation_model(data)
    
    # 2. GradCAM Setup - IGNORED as per request
    # grad_cam = GradCAM(model, criterion=torch.nn.CrossEntropyLoss())
    
    # 3. Generate Global Explanations
    print("Generating PGExplainer Global Map...")
    pg_global_imp = get_global_explanation(pgex, data)
    
    # print("Generating GradCAM Global Map...")
    # gc_global_imp = get_global_explanation(grad_cam, data)
    
    # 4. Evaluate metrics
    pg_metrics = evaluate_global_explanation(pg_global_imp, data, name="PGExplainer")
    # gc_metrics = evaluate_global_explanation(gc_global_imp, data, name="GradCAM")
    
    # 5. Correlation with Centralities
    print("Calculating Correlations with Centralities...")
    pg_corr, pg_pval = get_correlation_centralities(centralities, pg_global_imp.cpu().numpy(), data)
    pg_mi = get_mutual_information_centralities(centralities, pg_global_imp.cpu().numpy(), data)
    
    # gc_corr, gc_pval = get_correlation_centralities(centralities, gc_global_imp.cpu().numpy(), data)
    # gc_mi = get_mutual_information_centralities(centralities, gc_global_imp.cpu().numpy(), data)
    
    # Save metrics to file
    metrics_path = save_path.replace('.png', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("--- PGExplainer ---\n")
        f.write("Classification Metrics:\n")
        print(pg_metrics)
        for k, v in pg_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
            
        f.write("\nPearson Correlation with Centralities:\n")
        for k, v in pg_corr.items():
            f.write(f"{k}: {v:.4f} (p={pg_pval[k]:.4f})\n")
            
        f.write("\nMutual Information with Centralities:\n")
        for k, v in pg_mi.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Saved metrics to {metrics_path}")
    
    # 5. Visualize
    print(f"Saving global visualization to {save_path}...")
    
    # Prepare colors
    # GT
    gt_colors = (data.shape > 0).float().cpu().numpy()
    
    pos = dataset.G.pos if hasattr(dataset.G, 'pos') else None
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot GT
    dataset.visualize(show=False, ax=ax[0])
    ax[0].set_title("Ground Truth (Blue=Shape)")

    # Plot PGExplainer
    import networkx as nx
    G = dataset.G
    # Add importance as attribute
    for i, val in enumerate(pg_global_imp):
        G.nodes[i]['pg_imp'] = val.item()
        
    # Helper to draw
    def draw_imp(graph, attr, ax, title):
        # node_color = [graph.nodes[i][attr] for i in range(len(graph.nodes))]
        # Use simple nx draw
        colors = [graph.nodes[n][attr] for n in graph.nodes()]
        nx.draw(graph, pos=graph.pos if hasattr(graph, 'pos') else nx.spring_layout(graph, seed=42), 
                node_color=colors, cmap=plt.cm.Reds, node_size=50, ax=ax, with_labels=True)
        ax.set_title(title)

    draw_imp(G, 'pg_imp', ax[1], "PGExplainer Global Aggregation")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    base_save_dir = "/home/guilhermess/explanation-subgraphs/shapeggen"
    os.makedirs(base_save_dir, exist_ok=True)
    
    for i in range(10):
        seed = 42 + i
        run_name = f"run_seed_{seed}"
        run_dir = os.path.join(base_save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n================ STARTING {run_name} ================")
        
        dataset = get_dataset(seed=seed)
        
        # We also need to save the dataset visualization for this specific seed
        dataset_image_path = os.path.join(run_dir, "dataset.png")
        save_dataset_visualization(dataset, dataset_image_path)
        
        data = dataset.get_graph(use_fixed_split=True)
        # print(data)
        
        # Calculate and print class distribution (optional logging)
        # y = data.y
        # ... 

        dataset.num_classes = 2
        dataset.num_features = data.x.shape[1] 
        
        if not hasattr(data, 'val_mask') and hasattr(data, 'valid_mask'):
            data.val_mask = data.valid_mask

        # Train model for this dataset
        print(f"Training model for seed {seed}...")
        model, _ = optimize_hyperparameters(data, dataset)

        evaluate_model(model, data)
        
        explanation_image_path = os.path.join(run_dir, "explanation.png")
        run_global_explanations(model, dataset, data, explanation_image_path)
        print(f"================ FINISHED {run_name} ================\n")

if __name__ == "__main__":
    main()