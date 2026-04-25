import json

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "imports",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import torch.nn.functional as F\n",
                    "import networkx as nx\n",
                    "import matplotlib.pyplot as plt\n",
                    "from torch_geometric.loader import DataLoader\n",
                    "\n",
                    "from dataset_loader.load_dataset import load_molecule_datasets\n",
                    "from model.gcn import GraphGCN\n",
                    "from model.trainer import optimize_hyperparameters, get_model_checkpoint, save_model_checkpoint, evaluate_model\n",
                    "from explainer.gnn_explainer_wrapper import get_explainer\n",
                    "from metrics.fidelity import get_fidelity_metrics\n",
                    "\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "load_dataset",
                "metadata": {},
                "outputs": [],
                "source": [
                    "dataset_name = 'AlkaneCarbonyl'\n",
                    "data_dict = load_molecule_datasets(choice=dataset_name, split=(0.7, 0.2, 0.1))\n",
                    "\n",
                    "print(f'Dataset: {dataset_name}')\n",
                    "print(f\"Number of training graphs: {len(data_dict['train'][0])}\")\n",
                    "print(f\"Number of validation graphs: {len(data_dict['val'][0])}\")\n",
                    "print(f\"Number of test graphs: {len(data_dict['test'][0])}\")\n",
                    "print(f\"Num features: {data_dict['num_features']}\")\n",
                    "print(f\"Num classes: {data_dict['num_classes']}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "model_management",
                "metadata": {},
                "outputs": [],
                "source": [
                    "model_type = 'gcn_graph'\n",
                    "num_features = data_dict['num_features']\n",
                    "num_classes = data_dict['num_classes']\n",
                    "\n",
                    "# Load or train model\n",
                    "model, model_params = get_model_checkpoint(model_type, dataset_name, num_features, num_classes)\n",
                    "\n",
                    "if model is None:\n",
                    "    print('Checkpoint not found. Optimizing hyperparameters...')\n",
                    "    model, model_params = optimize_hyperparameters(data_dict, model_type=model_type)\n",
                    "    save_model_checkpoint(model, model_type, model_params, dataset_name)\n",
                    "else:\n",
                    "    print(f'Loaded checkpoint for {model_type} on {dataset_name}')\n",
                    "    print(f'Model params: {model_params}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "evaluate_test",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Evaluating model on test set...')\n",
                    "test_metrics = evaluate_model(model, data_dict, split='test')\n",
                    "print('\\nTest Set Metrics:')\n",
                    "for k, v in test_metrics.items():\n",
                    "    print(f'{k}: {v:.4f}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "visualize_setup",
                "metadata": {},
                "outputs": [],
                "source": [
                    "def visualize_graph(data, node_importance=None, edge_importance=None, title='Graph', threshold=0.5):\n",
                    "    G = nx.Graph()\n",
                    "    edge_index = data.edge_index.cpu().numpy()\n",
                    "    for i in range(edge_index.shape[1]):\n",
                    "        G.add_edge(edge_index[0, i], edge_index[1, i])\n",
                    "    \n",
                    "    pos = nx.spring_layout(G, seed=42)\n",
                    "    \n",
                    "    node_colors = []\n",
                    "    for i in range(data.num_nodes):\n",
                    "        if node_importance is not None and node_importance[i] > threshold:\n",
                    "            node_colors.append('red')\n",
                    "        else:\n",
                    "            node_colors.append('lightblue')\n",
                    "            \n",
                    "    edge_colors = []\n",
                    "    if edge_importance is not None:\n",
                    "        for u, v in G.edges():\n",
                    "            # Find edge index\n",
                    "            idx = -1\n",
                    "            for k in range(data.edge_index.shape[1]):\n",
                    "                if (data.edge_index[0, k] == u and data.edge_index[1, k] == v):\n",
                    "                    idx = k\n",
                    "                    break\n",
                    "            if idx != -1 and edge_importance[idx] > threshold:\n",
                    "                edge_colors.append('red')\n",
                    "            else:\n",
                    "                edge_colors.append('black')\n",
                    "    else:\n",
                    "        edge_colors = 'black'\n",
                    "        \n",
                    "    plt.figure(figsize=(8, 6))\n",
                    "    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=500, font_size=10)\n",
                    "    plt.title(title)\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "explanation_pipeline",
                "metadata": {},
                "outputs": [],
                "source": [
                    "from torch_geometric.explain import Explainer, GNNExplainer\n",
                    "\n",
                    "print('Setting up explainer...')\n",
                    "explainer = Explainer(\n",
                    "    model=model,\n",
                    "    algorithm=GNNExplainer(epochs=200),\n",
                    "    explanation_type='phenomenon',\n",
                    "    node_mask_type='attributes',\n",
                    "    edge_mask_type='object',\n",
                    "    model_config=dict(\n",
                    "        mode='multiclass_classification',\n",
                    "        task_level='graph',\n",
                    "        return_type='log_probs',\n",
                    "    ),\n",
                    ")\n",
                    "\n",
                    "# Pick a test graph\n",
                    "test_graphs, test_masks = data_dict['test']\n",
                    "idx = 0\n",
                    "sample_data = test_graphs[idx]\n",
                    "sample_label = sample_data.y\n",
                    "sample_gt = test_masks[idx]\n",
                    "\n",
                    "print(f'Generating explanation for graph {idx} (Label: {sample_label.item()})...')\n",
                    "explanation = explainer(sample_data.x, sample_data.edge_index, target=sample_label)\n",
                    "\n",
                    "node_mask = explanation.node_mask\n",
                    "if node_mask.dim() > 1: node_mask = node_mask.mean(dim=-1)\n",
                    "edge_mask = explanation.edge_mask\n",
                    "\n",
                    "print('\\nExplanation visualization:')\n",
                    "visualize_graph(sample_data, node_importance=node_mask, edge_importance=edge_mask, title=f'Explanation for Graph {idx}', threshold=0.2)\n",
                    "\n",
                    "print('Ground Truth visualization:')\n",
                    "visualize_graph(sample_data, node_importance=sample_gt, title=f'Ground Truth for Graph {idx}')\n",
                    "\n",
                    "# Calculate Fidelity\n",
                    "from metrics.fidelity import get_fidelity_metrics\n",
                    "# fidelity_metrics = get_fidelity_metrics(explanation, explainer) # Need to ensure wrapper compatibility\n",
                    "# print('Fidelity Metrics:', fidelity_metrics)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open('graph_prediction.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == '__main__':
    create_notebook()
    print('Created graph_prediction.ipynb')
