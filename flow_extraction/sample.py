import torch

dataset = torch.load(
    "outputs/pyg_dataset/graph_dataset.pt",
    weights_only=False
)

count = 0

for graph in dataset:

    if 1 in graph.y.tolist():

        count += 1

print("Graphs containing Malicious Nodes :", count)

print("Total Graphs :", len(dataset))