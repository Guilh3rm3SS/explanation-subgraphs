import torch
from dataset_loader.load_dataset import load_molecule_datasets
from model.trainer import optimize_hyperparameters, save_model_checkpoint

def verify_pipeline():
    print("Loading AlkaneCarbonyl dataset...")
    data_dict = load_molecule_datasets(choice="AlkaneCarbonyl", split=(0.7, 0.2, 0.1))
    
    print(f"Number of training graphs: {len(data_dict['train'][0])}")
    print(f"Number of validation graphs: {len(data_dict['val'][0])}")
    print(f"Number of test graphs: {len(data_dict['test'][0])}")
    print(f"Num features: {data_dict['num_features']}")
    print(f"Num classes: {data_dict['num_classes']}")
    
    print("\nStarting hyperparameter optimization for GraphGCN...")
    # Using small number of epochs for quick verification
    # I'll monkeypatch train_graph_level_model or just use small epochs if possible
    # Actually, train_graph_level_model has 500 epochs default. 
    # I'll pass epochs=2 for verification.
    
    import model.trainer
    original_train = model.trainer.train_graph_level_model
    def quick_train(*args, **kwargs):
        kwargs['epochs'] = 2
        return original_train(*args, **kwargs)
    
    model.trainer.train_graph_level_model = quick_train
    
    best_model, best_params = optimize_hyperparameters(data_dict, model_type="gcn_graph")
    
    print(f"\nBest params: {best_params}")
    
    save_model_checkpoint(best_model, "gcn_graph", best_params, "AlkaneCarbonyl")
    print("Verification script finished successfully!")

if __name__ == "__main__":
    verify_pipeline()
