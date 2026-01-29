from dataset_loader.load_dataset import load_dataset
import torch

dataset, data = load_dataset("CiteSeer")
print(f"Dataset: CiteSeer")
print(f"Is undirected: {data.is_undirected()}")
print(f"Edge index shape: {data.edge_index.shape}")
# Check if edges are symmetric
row, col = data.edge_index
mask = row < col
print(f"Number of edges (one direction): {mask.sum()}")
print(f"Total edges: {data.edge_index.size(1)}")
