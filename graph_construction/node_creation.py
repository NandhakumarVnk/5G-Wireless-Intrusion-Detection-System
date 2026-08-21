import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =====================================================
# NODE CREATION MODULE
# =====================================================

print("=" * 60)
print("NODE CREATION MODULE")
print("=" * 60)

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

WINDOW_SIZE = 5

# -----------------------------------------------------
# Load Progressive Window Dataset
# -----------------------------------------------------

input_file = f"outputs/progressive_windows/window_{WINDOW_SIZE}.csv"

df = pd.read_csv(input_file)

print(f"\nLoaded File : {input_file}")
print(f"Total Flow Records : {len(df):,}")

# -----------------------------------------------------
# Remove Non-Feature Columns
# -----------------------------------------------------

remove_columns = [
    "Flow_ID",
    "Window_ID",
    "Window_Size",
    "Label",
    "Attack Type",
    "Attack Tool"
]

feature_columns = [
    col for col in df.columns
    if col not in remove_columns
]

print(f"\nTotal Features Used : {len(feature_columns)}")

# -----------------------------------------------------
# Convert Features to Numeric
# -----------------------------------------------------

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# -----------------------------------------------------
# Handle Missing Values
# -----------------------------------------------------

nan_before = X.isnull().sum().sum()

X = X.fillna(0)

nan_after = X.isnull().sum().sum()

print(f"\nNaN Before Cleaning : {nan_before:,}")
print(f"NaN After Cleaning  : {nan_after:,}")

# -----------------------------------------------------
# Feature Normalization
# -----------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = X.astype(np.float32)

# -----------------------------------------------------
# Create Normalized DataFrame
# -----------------------------------------------------

normalized_df = pd.DataFrame(
    X,
    columns=feature_columns
)

# -----------------------------------------------------
# Preserve Metadata
# -----------------------------------------------------

normalized_df["Flow_ID"] = df["Flow_ID"].values

normalized_df["Window_ID"] = df["Window_ID"].values

normalized_df["Window_Size"] = df["Window_Size"].values

normalized_df["Label"] = df["Label"].values

normalized_df["Attack Type"] = df["Attack Type"].values

normalized_df["Attack Tool"] = df["Attack Tool"].values

# -----------------------------------------------------
# Save Normalized Node Features
# -----------------------------------------------------

os.makedirs(
    "outputs/node_features",
    exist_ok=True
)

output_file = (
    f"outputs/node_features/"
    f"node_features_window_{WINDOW_SIZE}.csv"
)

normalized_df.to_csv(
    output_file,
    index=False
)

print("\nNormalized Node Features Saved")
print(output_file)

# -----------------------------------------------------
# Create Graph Node Matrices
# -----------------------------------------------------

graph_nodes = {}

for window_id, group in normalized_df.groupby("Window_ID"):

    graph_nodes[int(window_id)] = group[
        feature_columns
    ].values.astype(np.float32)

print(f"\nTotal Graph Snapshots : {len(graph_nodes):,}")

# -----------------------------------------------------
# Display First Graph
# -----------------------------------------------------

first_graph = graph_nodes[1]

print("\nGraph Snapshot : 1")

print(f"Nodes    : {first_graph.shape[0]}")
print(f"Features : {first_graph.shape[1]}")

print("\nFirst Node Feature Vector")

print(first_graph[0])

# -----------------------------------------------------
# Dataset Information
# -----------------------------------------------------

print("\nSaved Columns")

print(normalized_df.columns.tolist())

print("\nNode Creation Completed Successfully.")

print("=" * 60)