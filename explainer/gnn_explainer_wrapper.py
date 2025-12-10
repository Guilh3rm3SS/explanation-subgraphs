import torch
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, GraphMaskExplainer, CaptumExplainer
from .subgraph_utils import add_relevant_indices
from captum.attr import IntegratedGradients

def get_explainer(model, data, node_index=None,
                         epochs=200, lr=0.01,
                         node_mask_type="attributes",
                         edge_mask_type="object",
                         num_layers=2,
                         explanation_type="model",
                         algorithm="gnnexplainer"):
    """
    Gera explicações para um nó usando o algoritmo especificado.
    algorithm pode ser: 'gnnexplainer', 'pgexplainer',
                        'graphmask', 'captum', 'dummy'
    """

    model.eval()

    # ---- Seleção do algoritmo ----
    if algorithm == "gnnexplainer":
        algo = GNNExplainer(epochs=epochs, lr=lr)

    elif algorithm == "pgexplainer":
        algo = PGExplainer(lr=lr, epochs=epochs)

    elif algorithm == "graphmask":
        algo = GraphMaskExplainer(model=model, num_layers=num_layers)

    elif algorithm == "captum":
        algo = CaptumExplainer(attribution_method=IntegratedGradients, n_steps=epochs)


    else:
        raise ValueError(f"Algoritmo '{algorithm}' não suportado.")

    # ---- Explainer wrapper ----
    explainer = Explainer(
        model=model,
        algorithm=algo,
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        )
    )

    # ---- Treinamento (apenas para PGExplainer) ----
    if algorithm == "pgexplainer":
        # PGExplainer needs to be trained
        for epoch in range(epochs):
            if hasattr(data, 'train_mask') and data.train_mask is not None:
                indices = data.train_mask.nonzero(as_tuple=True)[0]
            else:
                indices = torch.arange(data.num_nodes)
            
            for idx in indices:
                explainer.algorithm.train(epoch, model, data.x, data.edge_index, target=data.y, index=idx)



    # ---- Geração da explicação ----
    # explanation = explainer(data.x, data.edge_index, index=node_index)

    # explanation = add_relevant_indices(explanation)
    return explainer


# def get_explanation_node_binary(model, data, node_index=None,
#                          epochs=200, lr=0.1,
#                          node_mask_type="object",
#                          edge_mask_type="object"):

#     model.eval()

#     explainer = Explainer(
#         model=model,
#         algorithm=GNNExplainer(epochs=epochs, lr=lr),
#         explanation_type='model',
#         node_mask_type=node_mask_type,
#         edge_mask_type=edge_mask_type,
#         model_config=dict(
#             mode='multiclass_classification',
#             task_level='node',
#             return_type='log_probs',
#         )
#     )

#     explanation = explainer(data.x, data.edge_index, index=node_index)
#     explanation = add_relevant_indices(explanation)

#     if explanation.node_mask is not None:
#         node_mask = explanation.node_mask
#         if node_mask.dim() > 1:
#             node_mask = node_mask.sum(dim=1)
        
#         node_mask = (node_mask > 0).float()
#         explanation.node_mask = node_mask
    
#     edge_mask = explanation.edge_mask
#     edge_mask = (edge_mask > 0).float()
#     explanation.edge_mask = edge_mask

#     return explanation
