import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from torch_geometric.explain.metric import groundtruth_metrics
from explainer.gnn_explainer_wrapper import get_explainer
from explainer.importance_filters import filtered_node_importance
from metrics.fidelity import get_fidelity_metrics
from metrics.correlations import get_correlation_centralities, get_mutual_information_centralities

def run_local_explanation_pipeline(model, data, dataset_name, centralities, config):
    results = {}
    
    epochs = config.get("explainer_epochs", 500)
    lr = config.get("explainer_lr", 0.1)
    algorithm = config.get("algorithm", "gnnexplainer")
    explanation_type = config.get("explanation_type", "model")
    node_mask_type = config.get("node_mask_type", "attributes")
    num_layers = config.get("num_layers", 2)
    task_level = config.get("task_level", "node")
    
    target = data.y if explanation_type == "phenomenon" else None
    
    print(f"Running {algorithm} explainer...", flush=True)
    explainer = get_explainer(
        model, data,
        algorithm=algorithm, 
        explanation_type=explanation_type, 
        epochs=epochs, 
        lr=lr, 
        node_mask_type=node_mask_type, 
        num_layers=num_layers,
        task_level=task_level
    )
    
    explanation = explainer(data.x, data.edge_index, index=None, target=target)
    
    # 1. Fidelity
    fidelity_metrics = get_fidelity_metrics(explanation, explainer)
    print(f"{algorithm} Fidelity Metrics:", fidelity_metrics)
    results.update({f"{algorithm}_{k}": v for k, v in fidelity_metrics.items()})
    
    # 2. Centralities Correlation / MI
    if centralities is not None:
        node_imp = filtered_node_importance(explanation, data)
        corr, p_value = get_correlation_centralities(centralities, node_imp, data)
        results[f"{algorithm}_pearson"] = corr
        results[f"{algorithm}_pearson_pvalue"] = p_value
        
        mi = get_mutual_information_centralities(centralities, node_imp, data)
        results[f"{algorithm}_mi"] = mi
        
        print(f"\n========= {algorithm.capitalize()} vs Centralities Correlations =========")
        print("Pearson Correlation:")
        for k, v in corr.items():
            print(f"  {k}: {v:.4f} (p-value={p_value[k]:.4f})")
        print("\nMutual Information:")
        for k, v in mi.items():
            print(f"  {k}: {v:.4f}")
        print("===================================================================\n")

    # 3. Ground Truth Evaluation
    # For synthetic from main.py
    if dataset_name == "synthetic" and hasattr(data, "node_mask"):
        print("Calculating Ground Truth metrics...", flush=True)
        node_mask = explanation.node_mask
        if node_mask is not None and node_mask.dim() > 1:
            node_mask = node_mask.mean(dim=-1)
        
        gt_metrics = groundtruth_metrics(node_mask, data.node_mask)
        acc, recall, prec, f1, auc = gt_metrics
        results[f"{algorithm}_gt_accuracy"] = acc
        results[f"{algorithm}_gt_recall"] = recall
        results[f"{algorithm}_gt_precision"] = prec
        results[f"{algorithm}_gt_f1"] = f1
        results[f"{algorithm}_gt_auc"] = auc
        print("Ground Truth Metrics (synthetic):", gt_metrics)

    # For shapeggen manual testing mapping from main_synthetic.py
    elif dataset_name == "shapeggen":
        ground_explanation_mask = np.array(data.shape.cpu().numpy() > 0).astype(int)
        pred_explanation_mask = explanation.node_mask
        if pred_explanation_mask is not None and pred_explanation_mask.dim() > 1:
            pred_explanation_mask = pred_explanation_mask.mean(dim=-1)
            
        if pred_explanation_mask.max() > pred_explanation_mask.min():
            pred_explanation_mask = (pred_explanation_mask - pred_explanation_mask.min()) / (pred_explanation_mask.max() - pred_explanation_mask.min())
        
        y_pred_bin = (pred_explanation_mask.cpu().numpy() > 0.1).astype(int)
        
        auc = roc_auc_score(ground_explanation_mask, pred_explanation_mask.cpu().numpy())
        f1 = f1_score(ground_explanation_mask, y_pred_bin, zero_division=0.0)
        recall = recall_score(ground_explanation_mask, y_pred_bin, zero_division=0.0)
        precision = precision_score(ground_explanation_mask, y_pred_bin, zero_division=0.0)
        
        results[f"{algorithm}_shapeggen_auc"] = auc
        results[f"{algorithm}_shapeggen_f1"] = f1
        results[f"{algorithm}_shapeggen_recall"] = recall
        results[f"{algorithm}_shapeggen_precision"] = precision
        print(f"ShapeGGen Metrics - AUC: {auc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    return explanation, results
