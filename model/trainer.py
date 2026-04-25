import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, recall_score, precision_score
import copy
from model.gcn import GCN, GraphGCN

def train_one_model(data, num_features, num_classes, hidden_channels, num_layers, dropout, epochs=500, lr=0.1, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, num_classes, hidden_channels, num_layers, dropout).to(device)
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_model_state)
    print(f"Best validation loss: {best_val_loss}", flush=True)
    return model, best_val_loss

def train_gatv2conv_model(data, num_features, num_classes, hidden_channels, num_layers, dropout, epochs=500, lr=0.1, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATConv(in_channels=num_features, out_channels=hidden_channels, heads=num_layers, dropout=dropout).to(device)
    # model = GCN(num_features, num_classes, hidden_channels, num_layers, dropout).to(device)
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss
    
def train_graph_level_model(data_dict, num_features, num_classes, hidden_channels, num_layers, dropout, epochs=500, lr=0.1, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphGCN(num_features, hidden_channels, num_classes, num_layers, dropout).to(device)
    
    train_graphs, _ = data_dict["train"]
    val_graphs, _ = data_dict["val"]
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        scheduler.step()
        
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                val_loss += F.cross_entropy(out, batch.y).item() * batch.num_graphs
        
        val_loss /= len(val_graphs)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_model_state)
    print(f"Best validation loss: {best_val_loss}", flush=True)
    return model, best_val_loss

def evaluate_model(model, data):
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(data, dict) and "test" in data:
        # Graph classification
        test_graphs, _ = data["test"]
        loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                y_true.append(batch.y.cpu())
                y_pred.append(pred.cpu())
        
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        
    else:
        # Node classification
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()
        
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
        
    return f1, recall, precision

def optimize_hyperparameters(data, dataset=None, model_type="gcn"):
    if isinstance(data, dict):
        num_features = data["num_features"]
        num_classes = data["num_classes"]
    else:
        num_features = dataset.num_features
        num_classes = dataset.num_classes

    embeddings = [16, 32, 64, 128]
    dropouts = [0, 0.25, 0.5]
    layers = [2]
    # lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    lrs = [0.05]

    
    best_overall_val_loss = float('inf')
    best_params = {}
    best_model = None
    
    # print("Iniciando otimização de hiperparâmetros...")
    for n_layers in layers:
        for hidden in embeddings:
            for drop in dropouts:
                for lr in lrs:
                    # print(f"Testando: layers={n_layers}, hidden={hidden}, dropout={drop}, lr={lr}", flush=True)
                    if model_type == "gcn":
                        model, val_loss = train_one_model(data, num_features, num_classes, hidden, n_layers, drop, lr=lr)
                    elif model_type == "gcn_graph":
                        model, val_loss = train_graph_level_model(data, num_features, num_classes, hidden, n_layers, drop, lr=lr)
                    elif model_type == "gatconv":
                        model, val_loss = train_gatv2conv_model(data, num_features, num_classes, hidden, n_layers, drop, lr=lr)
                    
                    if val_loss < best_overall_val_loss:
                        best_overall_val_loss = val_loss
                        best_params = {'layers': n_layers, 'hidden': hidden, 'dropout': drop, 'lr': lr}
                        best_model = model
    
    print(f"Otimização finalizada. Melhores parâmetros: {best_params}")
    f1, recall, precision = evaluate_model(best_model, data)
    print(f"Métricas do modelo final - F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    return best_model, best_params


def save_model_checkpoint(model, model_type, model_params, dataset_name):
    import os
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{model_type}_{dataset_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params,
        'model_type': model_type
    }, path)
    print(f"Checkpoint salvo em {path}")

def get_model_checkpoint(model_type, dataset_name, num_features, num_classes):
    import os
    path = f"checkpoints/{model_type}_{dataset_name}.pt"
    if not os.path.exists(path):
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    model_params = checkpoint['model_params']
    
    if model_type == "gcn":
        model = GCN(num_features, num_classes, 
                    hidden_channels=model_params['hidden'], 
                    num_layers=model_params['layers'], 
                    dropout=model_params['dropout']).to(device)
    elif model_type == "gcn_graph":
        model = GraphGCN(num_features, 
                         hidden_channels=model_params['hidden'], 
                         num_classes=num_classes,
                         num_layers=model_params['layers'], 
                         dropout=model_params['dropout']).to(device)
    elif model_type == "gatconv":
        model = GATConv(in_channels=num_features, 
                        out_channels=model_params['hidden'], 
                        heads=model_params['layers'], 
                        dropout=model_params['dropout']).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint carregado de {path}")
    return model, model_params

