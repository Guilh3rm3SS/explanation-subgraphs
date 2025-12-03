import torch
from torch_geometric.explain import Explainer, GNNExplainer
from .subgraph_utils import add_relevant_indices

def get_explanation_node(model, data, node_index=None,
                         epochs=200, lr=0.1,
                         node_mask_type="object",
                         edge_mask_type="object"):

    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type='model',
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        )
    )

    explanation = explainer(data.x, data.edge_index, index=node_index)
    explanation = add_relevant_indices(explanation)
    return explanation

def get_explanation_node_binary(model, data, node_index=None,
                         epochs=200, lr=0.1,
                         node_mask_type="object",
                         edge_mask_type="object"):

    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type='model',
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        )
    )

    explanation = explainer(data.x, data.edge_index, index=node_index)
    explanation = add_relevant_indices(explanation)

    if explanation.node_mask is not None:
        node_mask = explanation.node_mask
        if node_mask.dim() > 1:
            node_mask = node_mask.sum(dim=1)
        
        node_mask = (node_mask > 0).float()
        explanation.node_mask = node_mask
    
    edge_mask = explanation.edge_mask
    edge_mask = (edge_mask > 0).float()
    explanation.edge_mask = edge_mask

    return explanation
