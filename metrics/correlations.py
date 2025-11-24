import numpy as np
from scipy.stats import pearsonr

def get_correlation_centralities(centralities, node_importance, subgraph):

    if hasattr(subgraph, "node_idx_original"):
        node_ids = subgraph.node_idx_original.tolist()
    else:
        node_ids = list(range(len(node_importance)))

    correlations = {}

    for name, values in centralities.items():
        aligned = np.array([values[nid] for nid in node_ids])
        corr, _ = pearsonr(aligned, node_importance)
        correlations[name] = corr

    return correlations
