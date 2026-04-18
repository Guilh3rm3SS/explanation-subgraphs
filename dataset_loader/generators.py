import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx


class SyntheticGraphGenerator:
    """
    Gerador de grafos sintéticos com injeção de Motifs configuráveis.
    Suporta: 'house' (5 nós), 'cycle' (6 nós), 'star' (6 nós).

    Features baseadas no paper BAShapes/GNNExplainer:
    - Nós BASE:  x ~ N(0, 1)
    - Nós MOTIF: x ~ N(feature_bias, 1)

    Conexão ao grafo base: cada nó do motif é conectado a um nó aleatório
    do grafo base (não só o nó 0), evitando que a GCN identifique o motif
    puramente pela posição estrutural isolada.

    feature_bias controla a dificuldade:
        2.0 → fácil   (distribuições bem separadas)
        1.0 → moderado  ← default recomendado
        0.5 → difícil
    """

    def __init__(self, num_nodes=300, num_houses=20, num_features=1,
                 motif_type='house', feature_bias=1.0):
        self.num_nodes = num_nodes
        self.num_motifs = num_houses
        self.num_features = num_features
        self.motif_type = motif_type.lower()
        self.feature_bias = feature_bias

    def _get_motif_structure(self):
        if self.motif_type == 'house':
            edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
            return edges, 5
        elif self.motif_type == 'cycle':
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
            return edges, 6
        elif self.motif_type == 'star':
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
            return edges, 6
        else:
            raise ValueError(f"Motif '{self.motif_type}' nao suportado. Use 'house', 'cycle' ou 'star'.")

    def generate(self):
        motif_edges, motif_size = self._get_motif_structure()
        total_nodes = self.num_nodes + motif_size * self.num_motifs

        # 1. Grafo base Barabási-Albert
        G = nx.barabasi_albert_graph(self.num_nodes, 5)

        labels = torch.zeros(total_nodes, dtype=torch.long)
        gt_edges = set()

        # 2. Injetar motifs com múltiplas conexões ao grafo base
        current_node = self.num_nodes
        for i in range(self.num_motifs):
            mapping = {j: current_node + j for j in range(motif_size)}

            for u_local, v_local in motif_edges:
                u, v = mapping[u_local], mapping[v_local]
                G.add_edge(u, v)
                gt_edges.add(tuple(sorted((u, v))))

            # CORREÇÃO: conectar 2 nós do motif ao grafo base (não só o nó 0)
            # Isso mistura os nós de motif estruturalmente com os nós base,
            # impedindo que a GCN os detecte só pela posição isolada.
            anchors = np.random.choice(motif_size, size=2, replace=False)
            for anchor in anchors:
                target = np.random.randint(0, self.num_nodes)
                G.add_edge(target, mapping[int(anchor)])

            labels[current_node: current_node + motif_size] = 1
            current_node += motif_size

        # 3. Converter para PyG
        data = from_networkx(G)

        # 4. Features: N(0,1) para base, N(feature_bias, 1) para motif
        #    num_features=1 por padrão — com 1 dimensão a sobreposição
        #    entre N(0,1) e N(bias,1) é real e não trivialmente separável.
        x = torch.randn((total_nodes, self.num_features))
        motif_mask = labels.bool()
        x[motif_mask] += self.feature_bias

        data.x = x
        data.y = labels

        # 5. Máscara GT de arestas
        edge_mask = torch.zeros(data.num_edges, dtype=torch.float)
        for idx, (u, v) in enumerate(data.edge_index.t().tolist()):
            if tuple(sorted((u, v))) in gt_edges:
                edge_mask[idx] = 1.0

        data.edge_mask_gt = edge_mask
        print(data)
        return data