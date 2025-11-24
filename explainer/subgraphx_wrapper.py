import torch
from dig.xgraph.method import SubgraphX

class SubgraphXExplainer:
    """
    Wrapper para o método SubgraphX (DIG) para explicação de modelos GNN.
    """

    def __init__(self,
                 model,
                 num_classes=None,
                 num_hops=None,
                 explain_graph=True,
                 rollout=20,
                 min_atoms=4,
                 c_puct=10.0,
                 expand_atoms=14,
                 high2low=False,
                 reward_method='gnn_score',
                 subgraph_building_method='zero_filling',
                 save_dir=None,
                 filename='explanation_results',
                 vis=False,
                 device=None):

        # Seleciona device automaticamente
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = model.to(self.device)
        self.explain_graph_mode = explain_graph

        # Se num_classes não foi passado, tenta extrair do modelo
        if num_classes is None:
            try:
                num_classes = model.out_channels
            except AttributeError:
                num_classes = 2  # default

        # Instancia o SubgraphX com todos os parâmetros
        self.explainer = SubgraphX(
            model=self.model,
            num_classes=num_classes,
            num_hops=num_hops,
            explain_graph=explain_graph,
            rollout=rollout,
            min_atoms=min_atoms,
            c_puct=c_puct,
            expand_atoms=expand_atoms,
            high2low=high2low,
            reward_method=reward_method,
            subgraph_building_method=subgraph_building_method,
            save_dir=save_dir,
            filename=filename,
            vis=vis,
            device=self.device
        )

    def explain_node(self, data, node_idx):
        """
        Explica um único nó do grafo.
        """
        data = data.to(self.device)

        explanation = self.explainer.explain(
            x=data.x,
            edge_index=data.edge_index,
            node_idx=node_idx
        )

        return explanation

    def explain_graph(self, data):
        """
        Explica o grafo inteiro (modo global).
        """
        data = data.to(self.device)

        # SubgraphX usa o mesmo método explain() para grafo
        # sem passar node_idx
        explanation = self.explainer.explain(
            x=data.x,
            edge_index=data.edge_index
        )

        return explanation