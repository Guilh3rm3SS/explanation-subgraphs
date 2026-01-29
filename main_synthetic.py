import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model.gcn import GCN
from model.trainer import optimize_hyperparameters
import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph, ERGraph
from torch_geometric.explain import Explainer, GNNExplainer
# from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
from torch.optim.lr_scheduler import CosineAnnealingLR

num_nodes = [300, 500]
edge_density = [0.3, 0.6, 0.9]
num_motifs = [20, 30]

for n_nodes in num_nodes:
    for edens in edge_density:
        for nmotifs in num_motifs:
            dataset = ExplainerDataset(
                # graph_generator=BAGraph(num_nodes=300, num_edges=299),
                graph_generator=ERGraph(num_nodes=n_nodes, edge_prob=edens),
                motif_generator='house',
                num_motifs=nmotifs,
                transform=T.Constant(),
            )
            data = dataset[0]
            print(data, flush=True)

            import copy

            # Create masks
            idx = torch.arange(data.num_nodes)
            train_idx, temp_idx = train_test_split(idx, train_size=0.6, stratify=data.y.cpu())
            val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, stratify=data.y[temp_idx].cpu())

            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = True
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = True
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = True

            # model, _ = optimize_hyperparameters(data, dataset, 'gcn')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.to(device)
            model = GCN(data.num_node_features, hidden_channels=20, num_layers=3,
                        num_classes=dataset.num_classes).to(device)

            epochs = 2000
            lr = 0.001
            patience = 200
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0

            print("Iniciando treinamento...", flush=True)
            pbar = tqdm(range(epochs))
            for epoch in pbar:
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                # Usando nll_loss pois o GCN retorna log_softmax
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
            scheduler.step()
    
            # Validation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
                
                # Accuracy for progress bar
                pred = out.argmax(dim=-1)
                train_acc = int((pred[data.train_mask] == data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
                val_acc = int((pred[data.val_mask] == data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            pbar.set_description(f'Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            
            if patience_counter >= patience:
                print(f"\nEarly stopping na época {epoch}", flush=True)
                break

            model.load_state_dict(best_model_state)
            print(f"Melhor perda de validação: {best_val_loss:.4f}", flush=True)

            @torch.no_grad()
            def test():
                model.eval()
                pred = model(data.x, data.edge_index).argmax(dim=-1)
                test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
                return test_acc

            final_test_acc = test()
            print(f"Acurácia final no teste: {final_test_acc:.4f}")
            model.eval()

            for explanation_type in ['phenomenon', 'model']:
                explainer = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=300),
                    explanation_type=explanation_type,
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='node',
                        return_type='raw',
                    ),
                )

                # Explanation ROC AUC over all test nodes:
                targets, preds = [], []
                start_idx = min(400, int(data.num_nodes * 0.8)) # Garante que comece dentro do range se o grafo for pequeno
                node_indices = range(start_idx, data.num_nodes, 5)
                
                # Se ainda estiver vazio, pega os últimos 10 nós
                if len(node_indices) == 0:
                    node_indices = range(max(0, data.num_nodes - 10), data.num_nodes)
                    
                for node_index in tqdm(node_indices, leave=False, desc='Train Explainer'):
                    target = data.y if explanation_type == 'phenomenon' else None
                    explanation = explainer(data.x, data.edge_index, index=node_index,
                                            target=target)

                    _, _, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops=3,
                                                            edge_index=data.edge_index)

                    targets.append(data.edge_mask[hard_edge_mask].cpu())
                    preds.append(explanation.edge_mask[hard_edge_mask].cpu())

                y_true = torch.cat(targets).numpy()
                y_pred_cont = torch.cat(preds).numpy()
                y_pred_bin = (y_pred_cont > 0.5).astype(int)

                auc = roc_auc_score(y_true, y_pred_cont)
                f1 = f1_score(y_true, y_pred_bin, zero_division=0.0)
                recall = recall_score(y_true, y_pred_bin, zero_division=0.0)
                precision = precision_score(y_true, y_pred_bin, zero_division=0.0)
        
                print(f'\n\nExplaining for {n_nodes} nodes, {edens} edge density and {nmotifs} motifs\n', flush=True)
                print(f'Mean ROC AUC (explanation type {explanation_type:10}): {auc:.4f}', flush=True)
                print(f'Mean F1 (explanation type {explanation_type:10}): {f1:.4f}', flush=True)
                print(f'Mean Recall (explanation type {explanation_type:10}): {recall:.4f}', flush=True)
                print(f'Mean Precision (explanation type {explanation_type:10}): {precision:.4f}', flush=True)
                