import torch
from dig.xgraph.method import SubgraphX
from torch_geometric.explain import Explanation

class GraphModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        # Global mean pooling to simulate graph classification output
        return out.mean(dim=0, keepdim=True)

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
        
        self.num_classes = num_classes

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
        
        # Calculate predicted label for the node
        logits = self.model(data.x, data.edge_index)
        if logits.dim() == 2:
            label = logits[node_idx].argmax(-1).item()
        else:
            # Handle batch dimension if present? Usually PyG models output [num_nodes, num_classes]
            label = logits[node_idx].argmax(-1).item()

        explanation = self.explainer.explain(
            x=data.x,
            edge_index=data.edge_index,
            node_idx=node_idx,
            label=label
        )

        return explanation

    def explain_graph(self, data, bin_weight=False):
        """
        Explica o grafo inteiro (modo global).
        """
        data = data.to(self.device)
        
        # SubgraphX usa o mesmo método explain() para grafo
        # sem passar node_idx
        
        # Wrap model to output graph-level prediction (mean of nodes)
        original_model = self.explainer.model
        self.explainer.model = GraphModelWrapper(self.model)
        
        aggregated_node_mask = torch.zeros((data.num_nodes, 1), device=self.device)
        aggregated_edge_mask = torch.ones(data.num_edges, device=self.device)
        
        for label in range(self.num_classes):
            explanation = self.explainer.explain(
                x=data.x,
                edge_index=data.edge_index,
                label=label,
                max_nodes=data.num_nodes,
                node_idx=None
            )
                
            # SubgraphX (DIG) returns a tuple (results, metrics)
            results = explanation[0]
            best_result = results[0]

            if label == 0:
                print(best_result)

            coalition = best_result['coalition']
            mask = torch.zeros((data.num_nodes, 1), device=self.device)
            if bin_weight == True:
                mask[coalition] = 1
            
            else:
                mask[coalition] = best_result['P']

            aggregated_node_mask += mask
                
                
        # Restore original model
        self.explainer.model = original_model
        
        # Create a single Explanation object
        final_explanation = Explanation(
            node_mask=aggregated_node_mask,
            edge_mask=aggregated_edge_mask,
            x=data.x,
            edge_index=data.edge_index
        )
        
        return final_explanation