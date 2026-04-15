import numpy as np
from scipy.stats import pearsonr

def get_correlation_centralities(centralities, node_importance, subgraph):

    if hasattr(subgraph, "node_idx_original"):
        node_ids = subgraph.node_idx_original.tolist()
    else:
        node_ids = list(range(len(node_importance)))

    correlations = {}
    p_values = {}

    for name, values in centralities.items():
        aligned = np.array([values[nid] for nid in node_ids])
        corr, p_val = pearsonr(aligned, node_importance)
        correlations[name] = corr
        p_values[name] = p_val

    return correlations, p_values

from sklearn.feature_selection import mutual_info_regression

def get_mutual_information_centralities(centralities, node_importance, subgraph):
    if hasattr(subgraph, "node_idx_original"):
        node_ids = subgraph.node_idx_original.tolist()
    else:
        node_ids = list(range(len(node_importance)))

    correlations = {}

    # Reshape node_importance for sklearn
    y = np.array(node_importance)
    
    for name, values in centralities.items():
        aligned = np.array([values[nid] for nid in node_ids]).reshape(-1, 1)
        # mutual_info_regression returns an array, we take the first element
        mi = mutual_info_regression(aligned, y)[0]
        correlations[name] = mi

    return correlations

