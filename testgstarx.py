import pandas as pd

from dataset_loader.load_dataset import load_dataset
from model.gcn import GCN
from explainer.gnn_explainer_wrapper import get_explanation_node
from explainer.subgraphx_wrapper import SubgraphXExplainer
from explainer.subgraph_utils import get_khop_subgraph
from explainer.importance_filters import filtered_node_importance
from metrics.centralities import get_centralities
from metrics.correlations import get_correlation_centralities
from GStarX.gstarx import GStarX

# ========= Load Dataset =========
dataset, data = load_dataset("KarateClub")

# ========= Build & train model =========
model = GCN(dataset)
optimizer = None

# Quick training
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Training model...")
for i in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


model.eval()
explainer = GStarX(model=model, device=device,
                    max_sample_size=10,
                    tau=0.01,
                    payoff_type="norm_prob",
                    payoff_avg=None,
                    subgraph_building_method="remove",
                   
                   )

explanation = explainer.explain(data,
                                node_idx=0,
                                superadditive_ext=True,
                                sample_method="khop",
                                num_samples=-1,
                                k=2
                                )
print(explanation)

# ========= Global Explanation =========
# print("\nGenerating explanation...")
# exp = get_explanation_node(model, data)
# subgraph = data

# centralities = get_centralities(subgraph)
# node_imp = filtered_node_importance(exp, subgraph)
# corr = get_correlation_centralities(centralities, node_imp, subgraph)

# df = pd.DataFrame([corr], index=["mean"])
# print("\n📊 Correlations:")
# print(df.round(4))
# df.to_csv("correlations_karateclub.csv")