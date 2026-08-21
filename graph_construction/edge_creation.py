import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

print("=" * 60)
print("EDGE CREATION MODULE")
print("=" * 60)

# =====================================================
# Configuration
# =====================================================

WINDOW_SIZE = 5

# Number of nearest neighbours
K = 2

# None = Process ALL windows
MAX_WINDOWS = None

# =====================================================
# Create Output Folder
# =====================================================

os.makedirs(
    "outputs/graph_edges",
    exist_ok=True
)

# =====================================================
# Load Progressive Window Dataset
# =====================================================

window_file = (
    f"outputs/progressive_windows/window_{WINDOW_SIZE}.csv"
)

print("\nLoading Dataset...")

df = pd.read_csv(
    window_file
)

print("Input File :", window_file)

print("Dataset Shape :", df.shape)

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

    for column in df.columns

    if column not in remove_columns

]

print("\n")
print("=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print("Total Features :", len(feature_columns))

print("\nFirst Five Features")

for feature in feature_columns[:5]:

    print("-", feature)

# =====================================================
# Global Feature Scaling
# =====================================================

print("\n")
print("=" * 60)
print("GLOBAL FEATURE SCALING")
print("=" * 60)

global_features = df[feature_columns]

global_features = global_features.apply(

    pd.to_numeric,

    errors="coerce"

)

global_features = global_features.fillna(0)

scaler = StandardScaler()

scaler.fit(global_features)

print("Global StandardScaler Fitted Successfully")

# =====================================================
# Group Dataset by Window
# =====================================================

graphs = df.groupby(

    "Window_ID"

)

total_windows = len(graphs)

print("\nTotal Windows :", total_windows)

graph_edges = {}

all_edges = []

processed = 0

print("\n")
print("=" * 60)
print("CREATING K-NN GRAPH EDGES")
print("=" * 60)
# =====================================================
# Create k-NN Graph Edges
# =====================================================

for window_id, group in graphs:

    # -------------------------------------------------
    # Process All Windows
    # -------------------------------------------------

    if MAX_WINDOWS is not None:

        if processed >= MAX_WINDOWS:

            break

    # -------------------------------------------------
    # Node Feature Matrix
    # -------------------------------------------------

    X = group[feature_columns]

    X = X.apply(

        pd.to_numeric,

        errors="coerce"

    )

    X = X.fillna(0)

    # -------------------------------------------------
    # Global Feature Scaling
    # -------------------------------------------------

    X = scaler.transform(X)

    # -------------------------------------------------
    # Build k-NN Graph
    # -------------------------------------------------

    nbrs = NearestNeighbors(

        n_neighbors=min(

            K + 1,

            len(group)

        ),

        metric="euclidean"

    )

    nbrs.fit(X)

    distances, indices = nbrs.kneighbors(X)

    edge_index = []

    edge_set = set()

    # -------------------------------------------------
    # Create Edges
    # -------------------------------------------------

    for source in range(len(indices)):

        for target in indices[source][1:]:

            # Remove self-loops

            if source == target:

                continue

            edge = (

                int(source),

                int(target)

            )

            # Remove duplicate edges

            if edge in edge_set:

                continue

            edge_set.add(edge)

            edge_index.append(

                [

                    int(source),

                    int(target)

                ]

            )

            all_edges.append(

                {

                    "Window_ID": int(window_id),

                    "Source": int(source),

                    "Target": int(target)

                }

            )

    # -------------------------------------------------
    # Store Graph Edge Index
    # -------------------------------------------------

    graph_edges[int(window_id)] = np.array(

        edge_index,

        dtype=np.int64

    )

    processed += 1

    # -------------------------------------------------
    # Progress
    # -------------------------------------------------

    if processed % 1000 == 0:

        print(

            f"{processed} Graphs Completed..."

        )
        # =====================================================
# Save Edge File
# =====================================================

edge_df = pd.DataFrame(all_edges)

edge_file = os.path.join(

    "outputs",

    "graph_edges",

    f"edges_window_{WINDOW_SIZE}.csv"

)

edge_df.to_csv(

    edge_file,

    index=False

)

# =====================================================
# Dataset Statistics
# =====================================================

print("\n")
print("=" * 60)
print("EDGE DATASET STATISTICS")
print("=" * 60)

print("Graphs Processed      :", processed)

print("Window Size           :", WINDOW_SIZE)

print("k (Nearest Neighbours):", K)

print("Total Edges           :", len(edge_df))

average_edges = (

    len(edge_df)

    /

    processed

)

print(f"Average Edges / Graph : {average_edges:.2f}")

# =====================================================
# Display First Graph
# =====================================================

print("\n")
print("=" * 60)
print("FIRST GRAPH")
print("=" * 60)

first_window = sorted(

    graph_edges.keys()

)[0]

first_graph = graph_edges[first_window]

print("Window ID :", first_window)

print("Nodes     :", len(

    df[df["Window_ID"] == first_window]

))

print("Edges     :", len(first_graph))

print("\nFirst 10 Edges")

print(first_graph[:10])

# =====================================================
# Verify Saved Edge File
# =====================================================

print("\n")
print("=" * 60)
print("VERIFY SAVED EDGE FILE")
print("=" * 60)

verify_df = pd.read_csv(edge_file)

print("Saved File Shape :", verify_df.shape)

print("\nFirst 10 Rows")

print(

    verify_df.head(10)

)

# =====================================================
# Save Completed
# =====================================================

print("\n")
print("=" * 60)
print("EDGE CREATION COMPLETED")
print("=" * 60)

print("Saved File :")

print(edge_file)

print("\nDataset Structure")

print("""

Window_ID | Source | Target

--------------------------------

1           0         1

1           0         3

1           1         2

1           2         4

...

""")

print("=" * 60)
print("GRAPH EDGE DATASET READY")
print("=" * 60)

print("Graphs Created :", processed)

print("Total Edges    :", len(edge_df))

print("Edge File      :", edge_file)

print("\n✔ Processed All Windows")

print("✔ Global Feature Scaling")

print("✔ k-NN Graph Construction")

print("✔ Duplicate Edges Removed")

print("✔ Self-loops Removed")

print("✔ Compatible with pyg_dataset.py")

print("\nReady for pyg_dataset.py")

print("=" * 60)