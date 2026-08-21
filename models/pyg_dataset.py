import os
import torch
import pandas as pd

from torch_geometric.data import Data

print("=" * 60)
print("PYTORCH GEOMETRIC DATASET CREATION")
print("=" * 60)

# =====================================================
# Configuration
# =====================================================

WINDOW_SIZE = 5

NODE_FILE = (
    f"outputs/node_features/node_features_window_{WINDOW_SIZE}.csv"
)

EDGE_FILE = (
    f"outputs/graph_edges/edges_window_{WINDOW_SIZE}.csv"
)

# =====================================================
# Load CSV Files
# =====================================================

print("\nLoading Files...\n")

node_df = pd.read_csv(
    NODE_FILE
)

edge_df = pd.read_csv(
    EDGE_FILE
)

print("Node Feature File :", NODE_FILE)
print("Edge File         :", EDGE_FILE)

print("\nNode Data Shape :", node_df.shape)
print("Edge Data Shape :", edge_df.shape)

# =====================================================
# Feature Columns
# =====================================================

remove_columns = [

    "Flow_ID",

    "Window_ID",

    "Window_Size",

    "Label",

    "Attack Type",

    "Attack Tool"

]

feature_columns = [

    column

    for column in node_df.columns

    if column not in remove_columns

]

print("\n")
print("=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print("Number of Node Features :", len(feature_columns))

print("\nFirst Five Features")

for feature in feature_columns[:5]:

    print(" -", feature)

# =====================================================
# Label Mapping
# =====================================================

label_map = {

    "Benign": 0,

    "Malicious": 1

}

print("\nLabel Mapping")

print(label_map)

# =====================================================
# Available Windows
# =====================================================

available_windows = sorted(

    edge_df["Window_ID"].unique()

)

print("\nTotal Windows :", len(available_windows))

# =====================================================
# Group Nodes by Window
# =====================================================

graphs = (

    node_df

    [

        node_df["Window_ID"].isin(

            available_windows

        )

    ]

    .groupby(

        "Window_ID"

    )

)

dataset = []

print("\n")
print("=" * 60)
print("CREATING PYTORCH GEOMETRIC GRAPHS")
print("=" * 60)
# =====================================================
# Create PyTorch Geometric Graphs
# =====================================================

print("\n")
print("=" * 60)
print("CREATING PYTORCH GEOMETRIC DATASET")
print("=" * 60)

for window_id, graph in graphs:

    # -------------------------------------------------
    # Node Features
    # -------------------------------------------------

    x = torch.tensor(

        graph[feature_columns].values,

        dtype=torch.float32

    )

    # -------------------------------------------------
    # Node Labels
    # -------------------------------------------------

    node_y = torch.tensor(

        graph["Label"]

        .map(label_map)

        .values,

        dtype=torch.long

    )

    # -------------------------------------------------
    # Graph Label
    # (1 if any node is malicious)
    # -------------------------------------------------

    graph_label = torch.tensor(

        [

            1 if torch.any(node_y == 1)

            else 0

        ],

        dtype=torch.long

    )

    # -------------------------------------------------
    # Edge Index
    # -------------------------------------------------

    graph_edges = edge_df[

        edge_df["Window_ID"] == window_id

    ]

    edge_index = torch.tensor(

        graph_edges[

            ["Source", "Target"]

        ].values.T,

        dtype=torch.long

    )

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    malicious_nodes = int(

        (node_y == 1).sum().item()

    )

    benign_nodes = int(

        (node_y == 0).sum().item()

    )

    # -------------------------------------------------
    # Create PyG Graph
    # -------------------------------------------------

    data = Data(

        x=x,

        edge_index=edge_index,

        y=node_y,                  # Node labels

        graph_y=graph_label,       # Graph label

        window_id=int(window_id),

        malicious_nodes=malicious_nodes,

        benign_nodes=benign_nodes

    )

    dataset.append(data)

    # -------------------------------------------------
    # Display First Three Graphs
    # -------------------------------------------------

    if len(dataset) <= 3:

        print("\n")
        print("=" * 50)
        print(f"GRAPH {len(dataset)}")
        print("=" * 50)

        print("Window ID          :", data.window_id)

        print("Nodes              :", data.num_nodes)

        print("Edges              :", data.num_edges)

        print("Feature Shape      :", data.x.shape)

        print("Edge Shape         :", data.edge_index.shape)

        print("Node Labels        :", data.y.tolist())

        print("Graph Label        :", data.graph_y.item())

        print("Malicious Nodes    :", data.malicious_nodes)

        print("Benign Nodes       :", data.benign_nodes)
# =====================================================
# Dataset Statistics
# =====================================================

print("\n")
print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

total_graphs = len(dataset)

benign_graphs = sum(

    graph.graph_y.item() == 0

    for graph in dataset

)

malicious_graphs = sum(

    graph.graph_y.item() == 1

    for graph in dataset

)

total_nodes = sum(

    graph.num_nodes

    for graph in dataset

)

total_edges = sum(

    graph.num_edges

    for graph in dataset

)

average_nodes = (

    total_nodes /

    total_graphs

)

average_edges = (

    total_edges /

    total_graphs

)

average_attack_nodes = (

    sum(

        graph.malicious_nodes

        for graph in dataset

    )

    /

    total_graphs

)

average_benign_nodes = (

    sum(

        graph.benign_nodes

        for graph in dataset

    )

    /

    total_graphs

)

print("Total Graphs            :", total_graphs)

print("Benign Graphs           :", benign_graphs)

print("Malicious Graphs        :", malicious_graphs)

print("Total Nodes             :", total_nodes)

print("Total Edges             :", total_edges)

print(f"\nAverage Nodes / Graph   : {average_nodes:.2f}")

print(f"Average Edges / Graph   : {average_edges:.2f}")

print(f"\nAverage Attack Nodes    : {average_attack_nodes:.2f}")

print(f"Average Benign Nodes    : {average_benign_nodes:.2f}")

# =====================================================
# Save Dataset
# =====================================================

save_dir = "outputs/pyg_dataset"

os.makedirs(

    save_dir,

    exist_ok=True

)

save_path = os.path.join(

    save_dir,

    "graph_dataset.pt"

)

torch.save(

    dataset,

    save_path

)

# =====================================================
# Save Completed
# =====================================================

print("\n")
print("=" * 60)
print("PYTORCH GEOMETRIC DATASET CREATED")
print("=" * 60)

print("Saved File :", save_path)

# =====================================================
# Dataset Structure
# =====================================================

print("\nDataset Structure\n")

print("""

Data(

    x = Node Feature Matrix,

    edge_index = Graph Connectivity,

    y = Node Labels,

    graph_y = Graph Label,

    window_id = Window ID,

    malicious_nodes = Number of Malicious Nodes,

    benign_nodes = Number of Benign Nodes

)

""")

print("=" * 60)

print("PYG DATASET READY FOR TEMPORAL SEQUENCE CREATION")

print("=" * 60)

print("Graphs Created :", total_graphs)

print("Feature Dimension :", len(feature_columns))

print("Node Labels : Available")

print("Graph Labels : Available")

print("Ready for sequence_dataset.py")

print("=" * 60)