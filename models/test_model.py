import torch

from htstcl_gnn import HTSTCL_GNN

print("=" * 60)
print("HTSTCL-GNN MODEL TEST")
print("=" * 60)

# ==========================================================
# Load Graph Dataset
# ==========================================================

dataset = torch.load(
    "outputs/pyg_dataset/graph_dataset.pt",
    weights_only=False
)

print(f"\nGraphs Loaded : {len(dataset)}")

# ==========================================================
# Display First Graph
# ==========================================================

graph = dataset[0]

print("\nFirst Graph")

print("Nodes :", graph.num_nodes)
print("Edges :", graph.num_edges)
print("Node Feature Shape :", graph.x.shape)
print("Edge Shape :", graph.edge_index.shape)
print("Node Labels :", graph.y.tolist())

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing Device :", DEVICE)

# ==========================================================
# Load Model
# ==========================================================

model = HTSTCL_GNN(

    input_dim=91,

    hidden_dim=128,

    embedding_dim=128,

    num_classes=2

).to(DEVICE)

model.eval()

# ==========================================================
# Create One Temporal Sequence
# ==========================================================

SEQUENCE_LENGTH = 5

graphs = []

sequence_label = 0

for i in range(SEQUENCE_LENGTH):

    g = dataset[i].to(DEVICE)

    graphs.append(g)

    if torch.any(g.y == 1):

        sequence_label = 1

sample = {

    "graphs": graphs,

    "label": sequence_label

}

batch_sequences = [sample]

print("\nSequence Length :", len(graphs))
print("Sequence Label  :", sequence_label)

# ==========================================================
# Forward Pass
# ==========================================================

with torch.no_grad():

    output = model(batch_sequences)

# ==========================================================
# Output Shapes
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL OUTPUT")
print("=" * 60)

print("\nLogits Shape")
print(output["logits"].shape)

print("\nEmbedding Shape")
print(output["embedding"].shape)

print("\nSpatial Embedding Shape")
print(output["spatial_embedding"].shape)

print("\nTemporal Embedding Shape")
print(output["temporal_embedding"].shape)

print("\nProjection Shape")
print(output["projection"].shape)

print("\nSpatial Weight Shape")
print(output["spatial_weight"].shape)

print("\nTemporal Weight Shape")
print(output["temporal_weight"].shape)

# ==========================================================
# Prediction
# ==========================================================

prediction = torch.argmax(

    output["logits"],

    dim=1

).item()

print("\nPredicted Label :", prediction)

print("Ground Truth    :", sequence_label)

# ==========================================================
# Prediction Confidence
# ==========================================================

probabilities = torch.softmax(

    output["logits"],

    dim=1

)

print("\nPrediction Probabilities")

print(probabilities.cpu().numpy())

# ==========================================================
# Attention Weights
# ==========================================================

print("\nSpatial Attention Weight")

print(

    output["spatial_weight"]

    .cpu()

    .numpy()

)

print("\nTemporal Attention Weight")

print(

    output["temporal_weight"]

    .cpu()

    .numpy()

)

# ==========================================================
# Test Completed
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL TEST COMPLETED SUCCESSFULLY")
print("=" * 60)