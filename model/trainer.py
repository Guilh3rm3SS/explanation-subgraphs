import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, recall_score, precision_score
import copy
from model.gcn import GCN

def train_one_model(data, num_features, num_classes, hidden_channels, num_layers, dropout, epochs=200, lr=0.1, patience=10):
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
            val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
            
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
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
            
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

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        
        f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        
    return f1, recall, precision

def optimize_hyperparameters(data, dataset, model_type="gcn"):
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    
    # Otimização finalizada. Melhores parâmetros: {'layers': 2, 'hidden': 32, 'dropout': 0.25, 'lr': 0.05}
    # Métricas do modelo final - F1: 0.8057, Recall: 0.8276, Precision: 0.7913



    embeddings = [16, 32, 64, 128]
    dropouts = [0, 0.25, 0.5]
    layers = [1, 2]
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    # embeddings = [32]
    # dropouts = [0.25]
    # layers = [2]
    # lrs = [0.05]
    
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

