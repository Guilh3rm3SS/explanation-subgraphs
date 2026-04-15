import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model.gcn import GCN
from model.trainer import optimize_hyperparameters
import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph, ERGraph
from torch_geometric.explain import Explainer, GNNExplainer
# from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_loader.load_dataset import load_dataset
from graphxai.datasets import ShapeGGen
import numpy as np
from metrics.fidelity import get_fidelity_metrics

num_nodes = [300, 500]
edge_density = [0.3, 0.6, 0.9]
num_motifs = [20, 30]


dataset, data = load_dataset("shapeggen")


model, _ = optimize_hyperparameters(data, dataset, 'gcn')

@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
    return test_acc

final_test_acc = test()
print(f"Acurácia final no teste: {final_test_acc:.4f}")
model.eval()

for explanation_type in ['phenomenon', 'model']:
    for node_mask_type in ['attributes', 'object']:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=300, lr=0.05),
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='probs',
            ),
        )

        ground_explanation_mask = np.array(data.shape.cpu().numpy() > 0).astype(int)
        # print(ground_explanation_mask)
        pred_explanation = explainer(
            data.x, 
            data.edge_index, 
            target=data.y if explanation_type == 'phenomenon' else None, 
            index=None
        )

        pred_explanation_mask = pred_explanation.node_mask
        if pred_explanation_mask is not None and pred_explanation_mask.dim() > 1:
            pred_explanation_mask = pred_explanation_mask.mean(dim=-1)

        if pred_explanation_mask.max() > pred_explanation_mask.min():
         pred_explanation_mask = (pred_explanation_mask - pred_explanation_mask.min()) / (pred_explanation_mask.max() - pred_explanation_mask.min())
        
        y_pred_bin = (pred_explanation_mask.cpu().numpy() > 0.1).astype(int)

        # print(pred_explanation_mask.cpu().numpy())
        # print(y_pred_bin)

        auc = roc_auc_score(ground_explanation_mask, pred_explanation_mask)
        f1 = f1_score(ground_explanation_mask, y_pred_bin, zero_division=0.0)
        recall = recall_score(ground_explanation_mask, y_pred_bin, zero_division=0.0)
        precision = precision_score(ground_explanation_mask, y_pred_bin, zero_division=0.0)

        print(f"\n\nAvaliando para {explanation_type} {node_mask_type}")
        fidelity_metrics = get_fidelity_metrics(pred_explanation, explainer)
        print(f'Fidelity metrics (explanation type {explanation_type:10}): {fidelity_metrics}', flush=True)
        # print(f'\n\nExplaining for {n_nodes} nodes, {edens} edge density and {nmotifs} motifs\n', flush=True)
        print(f'Mean ROC AUC (explanation type {explanation_type:10}): {auc:.4f}', flush=True)
        print(f'Mean F1 (explanation type {explanation_type:10}): {f1:.4f}', flush=True)
        print(f'Mean Recall (explanation type {explanation_type:10}): {recall:.4f}', flush=True)
        print(f'Mean Precision (explanation type {explanation_type:10}): {precision:.4f}', flush=True)
        