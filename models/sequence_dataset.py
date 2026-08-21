import os
import random
import torch

print("=" * 60)
print("TEMPORAL GRAPH SEQUENCE CREATION")
print("=" * 60)

# =====================================================
# Configuration
# =====================================================

SEQUENCE_LENGTH = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =====================================================
# Load Graph Dataset
# =====================================================

dataset = torch.load(
    "outputs/pyg_dataset/graph_dataset.pt",
    weights_only=False
)

print(f"\nTotal Graphs Loaded : {len(dataset)}")

# =====================================================
# Create Non-Overlapping Temporal Sequences
# =====================================================

sequence_dataset = []

print("\nCreating Temporal Sequences...\n")

for start_index in range(

    0,

    len(dataset),

    SEQUENCE_LENGTH

):

    # --------------------------------------------
    # Skip incomplete sequence
    # --------------------------------------------

    if start_index + SEQUENCE_LENGTH > len(dataset):

        break

    sequence = dataset[

        start_index:

        start_index + SEQUENCE_LENGTH

    ]

    malicious_graphs = 0
    benign_graphs = 0

    malicious_nodes = 0
    benign_nodes = 0

    # --------------------------------------------
    # Analyze Graphs
    # --------------------------------------------

    for graph in sequence:

        if graph.graph_y.item() == 1:

            malicious_graphs += 1

        else:

            benign_graphs += 1

        malicious_nodes += graph.malicious_nodes
        benign_nodes += graph.benign_nodes

    # --------------------------------------------
    # Sequence Label
    # If any graph is malicious,
    # the sequence is malicious.
    # --------------------------------------------

    sequence_label = (

        1

        if malicious_graphs > 0

        else 0

    )

    sequence_dataset.append(

        {

            "graphs": sequence,

            "label": sequence_label,

            "malicious_graphs": malicious_graphs,

            "benign_graphs": benign_graphs,

            "malicious_nodes": malicious_nodes,

            "benign_nodes": benign_nodes

        }

    )

print("\nSequence Length :", SEQUENCE_LENGTH)

print("Total Sequences :", len(sequence_dataset))
# =====================================================
# Display First Temporal Sequence
# =====================================================

print("\n")
print("=" * 60)
print("FIRST TEMPORAL SEQUENCE")
print("=" * 60)

first = sequence_dataset[0]

print("Sequence Label      :", first["label"])
print("Malicious Graphs    :", first["malicious_graphs"])
print("Benign Graphs       :", first["benign_graphs"])
print("Malicious Nodes     :", first["malicious_nodes"])
print("Benign Nodes        :", first["benign_nodes"])

graphs = first["graphs"]

for index, graph in enumerate(graphs):

    print(f"\nGraph {index + 1}")

    print("Window ID          :", graph.window_id)

    print("Graph Label        :", graph.graph_y.item())

    print("Nodes              :", graph.num_nodes)

    print("Edges              :", graph.num_edges)

    print("Node Feature Shape :", graph.x.shape)

    print("Node Labels        :", graph.y.tolist())

# =====================================================
# Dataset Statistics
# =====================================================

benign_sequences = sum(

    sequence["label"] == 0

    for sequence in sequence_dataset

)

malicious_sequences = sum(

    sequence["label"] == 1

    for sequence in sequence_dataset

)

average_malicious_graphs = (

    sum(

        sequence["malicious_graphs"]

        for sequence in sequence_dataset

    )

    /

    len(sequence_dataset)

)

average_benign_graphs = (

    sum(

        sequence["benign_graphs"]

        for sequence in sequence_dataset

    )

    /

    len(sequence_dataset)

)

average_malicious_nodes = (

    sum(

        sequence["malicious_nodes"]

        for sequence in sequence_dataset

    )

    /

    len(sequence_dataset)

)

average_benign_nodes = (

    sum(

        sequence["benign_nodes"]

        for sequence in sequence_dataset

    )

    /

    len(sequence_dataset)

)

print("\n")
print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print("Total Sequences       :", len(sequence_dataset))

print("Benign Sequences      :", benign_sequences)

print("Malicious Sequences   :", malicious_sequences)

print(f"\nAverage Malicious Graphs : {average_malicious_graphs:.2f}")

print(f"Average Benign Graphs    : {average_benign_graphs:.2f}")

print(f"\nAverage Malicious Nodes  : {average_malicious_nodes:.2f}")

print(f"Average Benign Nodes     : {average_benign_nodes:.2f}")

# =====================================================
# Shuffle Dataset
# =====================================================

print("\n")
print("=" * 60)
print("SHUFFLING DATASET")
print("=" * 60)

random.shuffle(sequence_dataset)

print("Dataset Shuffled Successfully")

print("No Downsampling Applied")

print("No Oversampling Applied")

print("Original Dataset Preserved")

print("Ready for Training")
# =====================================================
# Save Temporal Sequence Dataset
# =====================================================

save_dir = "outputs/sequence_dataset"

os.makedirs(

    save_dir,

    exist_ok=True

)

save_path = os.path.join(

    save_dir,

    "graph_sequences.pt"

)

torch.save(

    sequence_dataset,

    save_path

)

# =====================================================
# Save Completed
# =====================================================

print("\n")
print("=" * 60)
print("TEMPORAL GRAPH SEQUENCES CREATED")
print("=" * 60)

print("Saved File :", save_path)

# =====================================================
# Dataset Structure
# =====================================================

print("\nDataset Structure\n")

print("""

{

    "graphs": [

        Graph1,

        Graph2,

        Graph3,

        Graph4,

        Graph5

    ],

    "label": 0 or 1,

    "malicious_graphs": int,

    "benign_graphs": int,

    "malicious_nodes": int,

    "benign_nodes": int

}

""")

# =====================================================
# Final Summary
# =====================================================

print("=" * 60)
print("SEQUENCE DATASET READY FOR HTSTCL-GNN TRAINING")
print("=" * 60)

print("Sequence Length        :", SEQUENCE_LENGTH)

print("Total Sequences        :", len(sequence_dataset))

print("Benign Sequences       :", benign_sequences)

print("Malicious Sequences    :", malicious_sequences)

print("\nDataset Information")

print("✔ Original Dataset Preserved")

print("✔ Non-overlapping Sequences")

print("✔ No Downsampling")

print("✔ No Random Undersampling")

print("✔ No Oversampling")

print("✔ Graph Labels Available")

print("✔ Node Labels Available")

print("✔ Suitable for Weighted CrossEntropyLoss")

print("✔ Suitable for Contrastive Learning")

print("\nReady for train.py")

print("=" * 60)