import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from graphxai.explainers import PGExplainer
from metrics.centralities import get_centralities
from metrics.correlations import get_correlation_centralities, get_mutual_information_centralities

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

def run_global_explanations(model, dataset, data, save_path, pg_epochs=100, pg_lr=0.003):
    print("Running GLOBAL explainers...")
    
    # Calculate Centralities (Degree, Betweenness, etc.)
    print("Calculating Graph Centralities...")
    centralities = get_centralities(data)
    
    # PGExplainer
    # We want the layer PRODUCING embeddings. Usually the one before the classifier.
    # In GCN, this is the second to last layer (index -2).
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
    pgex = PGExplainer(model, emb_layer_name='explanation_emb_layer', in_channels=2 * target_layer.out_channels, max_epochs=pg_epochs, lr=pg_lr)
    pgex.train_explanation_model(data)
    
    # 3. Generate Global Explanations
    print("Generating PGExplainer Global Map...")
    pg_global_imp = get_global_explanation(pgex, data)
    
    # 4. Evaluate metrics
    pg_metrics = evaluate_global_explanation(pg_global_imp, data, name="PGExplainer")
    
    # 5. Correlation with Centralities
    print("Calculating Correlations with Centralities...")
    pg_corr, pg_pval = get_correlation_centralities(centralities, pg_global_imp.cpu().numpy(), data)
    pg_mi = get_mutual_information_centralities(centralities, pg_global_imp.cpu().numpy(), data)
    
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
    G = dataset.G
    # Add importance as attribute
    for i, val in enumerate(pg_global_imp):
        G.nodes[i]['pg_imp'] = val.item()
        
    # Helper to draw
    def draw_imp(graph, attr, ax, title):
        colors = [graph.nodes[n][attr] for n in graph.nodes()]
        nx.draw(graph, pos=graph.pos if hasattr(graph, 'pos') else nx.spring_layout(graph, seed=42), 
                node_color=colors, cmap=plt.cm.Reds, node_size=50, ax=ax, with_labels=True)
        ax.set_title(title)

    draw_imp(G, 'pg_imp', ax[1], "PGExplainer Global Aggregation")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
