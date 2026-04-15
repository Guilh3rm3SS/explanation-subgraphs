import torch
from torch_geometric.explain import Explanation
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

def add_relevant_indices(explanation: Explanation) -> Explanation:
    out = explanation.clone()

    node_mask = explanation.get("node_mask")
    edge_mask = explanation.get("edge_mask")

    if node_mask is not None:
        node_relevance = node_mask.sum(dim=-1) > 0
        out.relevant_nodes = node_relevance.nonzero(as_tuple=True)[0]
    else:
        out.relevant_nodes = torch.tensor([], dtype=torch.long)

    if edge_mask is not None:
        edge_relevance = edge_mask > 0
        out.relevant_edges = edge_relevance.nonzero(as_tuple=True)[0]
    else:
        out.relevant_edges = torch.tensor([], dtype=torch.long)

    return out


def get_khop_subgraph(data, node_idx, num_hops=2, relabel_nodes=False):

    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=data.num_nodes,
    )

    x_sub = data.x[subset]
    y_sub = data.y[subset]

    ds = Data(
        x=x_sub,
        edge_index=edge_index_sub,
        y=y_sub
    )

    ds.node_idx_original = subset
    ds.center_mapping = mapping
    return ds
