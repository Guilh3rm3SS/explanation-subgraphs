import torch
import numpy as np

def filtered_node_importance(explanation, subgraph):
    if explanation.node_mask.dim() == 1:
        node_importance = explanation.node_mask
    else:
        node_importance = explanation.node_mask.sum(dim=1)

    if hasattr(subgraph, "node_idx_original"):
        return node_importance[subgraph.node_idx_original].cpu().numpy()

    return node_importance.cpu().numpy()


def filtered_edge_importance(explanation, subgraph):
    full_edges = explanation.edge_index.t()
    sub_edges = subgraph.edge_index.t()

    mask = torch.zeros(full_edges.size(0), dtype=torch.bool, device=full_edges.device)

    for e in sub_edges:
        mask |= (full_edges == e).all(dim=1)

    return explanation.edge_mask[mask].cpu().numpy()
