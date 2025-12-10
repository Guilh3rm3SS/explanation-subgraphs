import networkx as nx
from torch_geometric.utils import to_networkx

def to_networkx_fixed(data, to_undirected=True):
    if hasattr(data, "node_idx_original"):
        mapping = data.node_idx_original.tolist()
    else:
        mapping = list(range(data.num_nodes))

    G = to_networkx(data, to_undirected=to_undirected)
    return nx.relabel_nodes(G, {i: mapping[i] for i in range(len(mapping))})


def get_centralities(data):
    G = to_networkx_fixed(data, to_undirected=False)

    centralities = {
        "closeness": nx.closeness_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "degree": nx.degree_centrality(G),
        "in_degree": nx.in_degree_centrality(G),
        "out_degree": nx.out_degree_centrality(G),
        "eigenvector": nx.eigenvector_centrality(G, weight="weight"),
        "pagerank": nx.pagerank(G, weight="weight"),
    }

    return centralities
