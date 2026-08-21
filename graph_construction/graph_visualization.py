import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# GRAPH VISUALIZATION MODULE
# 5G WIRELESS IDS - TEMPORAL GRAPH CONSTRUCTION
# ============================================================

print("=" * 70)
print("5G WIRELESS IDS - GRAPH VISUALIZATION MODULE")
print("=" * 70)

# ============================================================
# 1. CONFIGURATION
# ============================================================

WINDOW_SIZE = 5

# Full dataset is used for statistics.
# Only first 20 graphs are visualized.
MAX_GRAPHS_TO_VISUALIZE = 20

K_NEIGHBORS = 2

WINDOW_FILE = (
    f"outputs/progressive_windows/window_{WINDOW_SIZE}.csv"
)

EDGE_FILE = (
    f"outputs/graph_edges/edges_window_{WINDOW_SIZE}.csv"
)

OUTPUT_DIR = "outputs/graph_visualization"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n")
print("=" * 70)
print("LOADING GRAPH CONSTRUCTION DATA")
print("=" * 70)

print("\nWindow File :")
print(WINDOW_FILE)

print("\nEdge File :")
print(EDGE_FILE)

window_df = pd.read_csv(
    WINDOW_FILE
)

edge_df = pd.read_csv(
    EDGE_FILE
)

print("\nDataset Information")
print("-" * 50)

print(
    "Window Dataset Shape :",
    window_df.shape
)

print(
    "Edge Dataset Shape   :",
    edge_df.shape
)

# ============================================================
# 3. DETERMINE GRAPH INFORMATION
# ============================================================

available_graphs = sorted(
    window_df["Window_ID"].unique()
)

total_graphs = len(
    available_graphs
)

print("\n")
print("=" * 70)
print("FULL GRAPH DATASET INFORMATION")
print("=" * 70)

print(
    "Total Graph Snapshots :",
    total_graphs
)

print(
    "Window Size           :",
    WINDOW_SIZE
)

print(
    "Nodes per Window      :",
    window_df.groupby("Window_ID").size().mean()
)

print(
    "Total Flow Records    :",
    len(window_df)
)

print(
    "Total Graph Edges     :",
    len(edge_df)
)

print(
    "k-Nearest Neighbours  :",
    K_NEIGHBORS
)

print(
    "Features per Node     :",
    91
)

# ============================================================
# 4. GLOBAL GRAPH STATISTICS
# ============================================================

nodes_per_graph = (
    window_df
    .groupby("Window_ID")
    .size()
)

edges_per_graph = (
    edge_df
    .groupby("Window_ID")
    .size()
)

print("\n")
print("=" * 70)
print("GRAPH DATASET STATISTICS")
print("=" * 70)

print(
    "Average Nodes / Graph :",
    f"{nodes_per_graph.mean():.2f}"
)

print(
    "Minimum Nodes / Graph:",
    nodes_per_graph.min()
)

print(
    "Maximum Nodes / Graph:",
    nodes_per_graph.max()
)

print(
    "Average Edges / Graph:",
    f"{edges_per_graph.mean():.2f}"
)

print(
    "Minimum Edges / Graph:",
    edges_per_graph.min()
)

print(
    "Maximum Edges / Graph:",
    edges_per_graph.max()
)

# ============================================================
# 5. LABEL INFORMATION
# ============================================================

if "Label" in window_df.columns:

    benign_records = (
        window_df["Label"]
        .astype(str)
        .str.lower()
        .eq("benign")
        .sum()
    )

    malicious_records = (
        window_df["Label"]
        .astype(str)
        .str.lower()
        .eq("malicious")
        .sum()
    )

    print("\n")
    print("=" * 70)
    print("NODE LABEL INFORMATION")
    print("=" * 70)

    print(
        "Benign Flow Nodes    :",
        benign_records
    )

    print(
        "Malicious Flow Nodes :",
        malicious_records
    )

# ============================================================
# 6. SELECT FIRST 20 GRAPHS FOR VISUALIZATION
# ============================================================

graphs_to_visualize = available_graphs[
    :MAX_GRAPHS_TO_VISUALIZE
]

print("\n")
print("=" * 70)
print("GRAPH VISUALIZATION CONFIGURATION")
print("=" * 70)

print(
    "Total Graphs Available :",
    total_graphs
)

print(
    "Graphs Visualized      :",
    len(graphs_to_visualize)
)

print(
    "Graphs Not Visualized  :",
    total_graphs - len(graphs_to_visualize)
)

print(
    "\nNote:"
)

print(
    "The complete dataset contains all",
    total_graphs,
    "graph snapshots."
)

print(
    "Only the first",
    MAX_GRAPHS_TO_VISUALIZE,
    "graphs are visualized to keep the output manageable."
)

# ============================================================
# 7. GRAPH SUMMARY STORAGE
# ============================================================

graph_summary = []

# ============================================================
# 8. BUILD AND VISUALIZE GRAPHS
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING AND VISUALIZING GRAPH SNAPSHOTS")
print("=" * 70)

for graph_index, graph_id in enumerate(
    graphs_to_visualize,
    start=1
):

    # --------------------------------------------------------
    # Extract nodes belonging to current temporal window
    # --------------------------------------------------------

    graph_window = window_df[
        window_df["Window_ID"] == graph_id
    ].copy()

    # --------------------------------------------------------
    # Extract edges belonging to current graph
    # --------------------------------------------------------

    graph_edges = edge_df[
        edge_df["Window_ID"] == graph_id
    ].copy()

    if graph_window.empty:
        continue

    # --------------------------------------------------------
    # Create Directed Graph
    #
    # k-NN creates a directed relationship:
    #
    # Node A -> nearest Node B
    #
    # Node B -> nearest Node C
    #
    # Reciprocal relationships are kept separately.
    # --------------------------------------------------------

    G = nx.DiGraph()

    # --------------------------------------------------------
    # Add Nodes
    # --------------------------------------------------------

    for node_index in range(
        len(graph_window)
    ):

        row = graph_window.iloc[node_index]

        node_label = str(
            row.get(
                "Label",
                "Unknown"
            )
        )

        G.add_node(
            node_index,
            label=node_label,
            duration=row.get(
                "Dur",
                0
            ),
            packets=row.get(
                "TotPkts",
                0
            ),
            rate=row.get(
                "Rate",
                0
            )
        )

    # --------------------------------------------------------
    # Add Edges
    # --------------------------------------------------------

    for _, row in graph_edges.iterrows():

        source = int(
            row["Source"]
        )

        target = int(
            row["Target"]
        )

        G.add_edge(
            source,
            target
        )

    # ========================================================
    # GRAPH LABEL
    # ========================================================

    malicious_nodes = 0
    benign_nodes = 0

    for node in G.nodes():

        label = str(
            G.nodes[node]["label"]
        ).lower()

        if label == "malicious":
            malicious_nodes += 1

        elif label == "benign":
            benign_nodes += 1

    if malicious_nodes > 0:
        graph_label = "Malicious"
    else:
        graph_label = "Benign"

    # ========================================================
    # SAVE GRAPH SUMMARY
    # ========================================================

    graph_summary.append(
        {
            "Window_ID": int(graph_id),
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Features_Per_Node": 91,
            "K_Neighbors": K_NEIGHBORS,
            "Benign_Nodes": benign_nodes,
            "Malicious_Nodes": malicious_nodes,
            "Graph_Label": graph_label
        }
    )

    # ========================================================
    # PRINT FIRST 20 GRAPH INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        f"GRAPH SNAPSHOT {graph_index} / "
        f"{len(graphs_to_visualize)}"
    )
    print("=" * 70)

    print(
        "Window ID          :",
        graph_id
    )

    print(
        "Graph Label        :",
        graph_label
    )

    print(
        "Nodes              :",
        G.number_of_nodes()
    )

    print(
        "Edges              :",
        G.number_of_edges()
    )

    print(
        "Features / Node    :",
        91
    )

    print(
        "k-NN               :",
        K_NEIGHBORS
    )

    print(
        "Benign Nodes       :",
        benign_nodes
    )

    print(
        "Malicious Nodes    :",
        malicious_nodes
    )

    # ========================================================
    # FIGURE
    # ========================================================

    plt.figure(
        figsize=(12, 9)
    )

    ax = plt.gca()

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.5
    )

    # ========================================================
    # NODE COLORS
    # ========================================================

    node_colors = []

    for node in G.nodes():

        label = str(
            G.nodes[node]["label"]
        ).lower()

        if label == "malicious":

            node_colors.append(
                "tomato"
            )

        else:

            node_colors.append(
                "mediumseagreen"
            )

    # ========================================================
    # DRAW EDGES
    # ========================================================

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=2,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.08"
    )

    # ========================================================
    # DRAW NODES
    # ========================================================

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=2200,
        node_color=node_colors,
        edgecolors="black",
        linewidths=2
    )

    # ========================================================
    # NODE IDENTIFIERS
    # ========================================================

    node_labels = {}

    for node in G.nodes():

        node_labels[node] = (
            f"F{node + 1}"
        )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        ax=ax,
        font_size=12,
        font_weight="bold",
        font_color="white"
    )

    # ========================================================
    # NODE INFORMATION
    # ========================================================

    for node in G.nodes():

        x, y = pos[node]

        node_data = G.nodes[node]

        duration = node_data["duration"]
        packets = node_data["packets"]
        rate = node_data["rate"]
        label = node_data["label"]

        try:
            duration_text = f"{float(duration):.3f}"
        except:
            duration_text = str(duration)

        try:
            packets_text = f"{float(packets):.0f}"
        except:
            packets_text = str(packets)

        try:
            rate_text = f"{float(rate):.2f}"
        except:
            rate_text = str(rate)

        info_text = (
            f"Label: {label}\n"
            f"Dur: {duration_text}\n"
            f"Pkts: {packets_text}\n"
            f"Rate: {rate_text}"
        )

        plt.text(
            x,
            y - 0.18,
            info_text,
            fontsize=8,
            ha="center",
            va="top",
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                alpha=0.95
            )
        )

    # ========================================================
    # GRAPH INFORMATION BOX
    # ========================================================

    info = (
        f"WINDOW / GRAPH INFORMATION\n"
        f"--------------------------------\n"
        f"Window ID        : {graph_id}\n"
        f"Nodes            : {G.number_of_nodes()}\n"
        f"Edges            : {G.number_of_edges()}\n"
        f"Features / Node  : 91\n"
        f"k-NN             : {K_NEIGHBORS}\n"
        f"Benign Nodes     : {benign_nodes}\n"
        f"Malicious Nodes  : {malicious_nodes}\n"
        f"Graph Label      : {graph_label}\n"
        f"--------------------------------\n"
        f"Node = Network Flow\n"
        f"Edge = k-NN Relationship"
    )

    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            facecolor="lightyellow",
            edgecolor="black",
            alpha=0.95
        )
    )

    # ========================================================
    # LEGEND
    # ========================================================

    legend_elements = [

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="mediumseagreen",
            markeredgecolor="black",
            markersize=12,
            label="Benign Flow Node"
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tomato",
            markeredgecolor="black",
            markersize=12,
            label="Malicious Flow Node"
        ),

        Line2D(
            [0],
            [0],
            color="gray",
            lw=2,
            label="k-NN Directed Edge"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9
    )

    # ========================================================
    # TITLE
    # ========================================================

    plt.title(
        (
            f"5G Wireless IDS - Temporal Graph Snapshot {graph_id}\n"
            f"Window {graph_id} | "
            f"5 Nodes | {G.number_of_edges()} k-NN Edges | "
            f"91 Features/Node"
        ),
        fontsize=15,
        fontweight="bold"
    )

    plt.axis("off")

    plt.tight_layout()

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    image_path = os.path.join(
        OUTPUT_DIR,
        f"graph_snapshot_{graph_id}.png"
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graph {graph_id} saved -> {image_path}"
    )

# ============================================================
# 9. SAVE GRAPH SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame(
    graph_summary
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "first_20_graph_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

# ============================================================
# 10. CREATE 20-GRAPH OVERVIEW
# ============================================================

print("\n")
print("=" * 70)
print("CREATING 20-GRAPH OVERVIEW")
print("=" * 70)

fig, axes = plt.subplots(
    4,
    5,
    figsize=(20, 15)
)

axes = axes.flatten()

for plot_index, graph_id in enumerate(
    graphs_to_visualize
):

    graph_window = window_df[
        window_df["Window_ID"] == graph_id
    ]

    graph_edges = edge_df[
        edge_df["Window_ID"] == graph_id
    ]

    G = nx.DiGraph()

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    for node_index in range(
        len(graph_window)
    ):

        row = graph_window.iloc[node_index]

        G.add_node(
            node_index,
            label=str(
                row.get(
                    "Label",
                    "Unknown"
                )
            )
        )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    for _, row in graph_edges.iterrows():

        G.add_edge(
            int(row["Source"]),
            int(row["Target"])
        )

    ax = axes[plot_index]

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.2
    )

    colors = []

    for node in G.nodes():

        label = str(
            G.nodes[node]["label"]
        ).lower()

        if label == "malicious":

            colors.append(
                "tomato"
            )

        else:

            colors.append(
                "mediumseagreen"
            )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        width=1.3,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.08"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=550,
        node_color=colors,
        edgecolors="black"
    )

    labels = {
        node: f"F{node + 1}"
        for node in G.nodes()
    }

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=7,
        font_weight="bold",
        font_color="white"
    )

    ax.set_title(
        f"Graph {graph_id}",
        fontsize=10,
        fontweight="bold"
    )

    ax.axis("off")

# ------------------------------------------------------------
# Overall Title
# ------------------------------------------------------------

fig.suptitle(
    (
        "5G Wireless IDS - First 20 Temporal Graph Snapshots\n"
        "Each Node = Network Flow | Each Edge = k-NN Relationship | k = 2"
    ),
    fontsize=18,
    fontweight="bold"
)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="mediumseagreen",
        markeredgecolor="black",
        markersize=10,
        label="Benign Node"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="tomato",
        markeredgecolor="black",
        markersize=10,
        label="Malicious Node"
    ),

    Line2D(
        [0],
        [0],
        color="gray",
        lw=2,
        label="Directed k-NN Edge"
    )
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=11
)

plt.tight_layout(
    rect=[0, 0.04, 1, 0.95]
)

overview_file = os.path.join(
    OUTPUT_DIR,
    "first_20_graph_overview.png"
)

plt.savefig(
    overview_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Overview Saved :",
    overview_file
)

# ============================================================
# 11. VERIFY OUTPUT IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("VERIFYING GRAPH VISUALIZATION OUTPUT")
print("=" * 70)

image_files = [

    file

    for file in os.listdir(
        OUTPUT_DIR
    )

    if file.endswith(".png")
]

print(
    "Individual Graph Images :",
    len(
        [
            f
            for f in image_files
            if f.startswith("graph_snapshot_")
        ]
    )
)

print(
    "Overview Image           :",
    "first_20_graph_overview.png"
    if "first_20_graph_overview.png" in image_files
    else "Missing"
)

print(
    "Summary CSV              :",
    summary_file
)

# ============================================================
# 12. FINAL DATASET SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("GRAPH VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFULL DATASET")
print("-" * 50)

print(
    "Total Graph Snapshots :",
    total_graphs
)

print(
    "Total Flow Nodes      :",
    len(window_df)
)

print(
    "Total k-NN Edges      :",
    len(edge_df)
)

print(
    "Features per Node     :",
    91
)

print(
    "Nodes per Graph       :",
    f"{nodes_per_graph.mean():.2f}"
)

print(
    "Edges per Graph       :",
    f"{edges_per_graph.mean():.2f}"
)

print("\nVISUALIZATION")
print("-" * 50)

print(
    "Graphs Visualized     :",
    len(graphs_to_visualize)
)

print(
    "Images Generated      :",
    len(
        [
            f
            for f in image_files
            if f.startswith("graph_snapshot_")
        ]
    )
)

print(
    "Output Folder         :",
    OUTPUT_DIR
)

# ============================================================
# 13. EXPLANATION FOR PROJECT DEMONSTRATION
# ============================================================

print("\n")
print("=" * 70)
print("GRAPH CONSTRUCTION EXPLANATION")
print("=" * 70)

print(
    """
1. Each temporal window represents one graph snapshot.

2. Each flow record inside a window becomes one graph node.

3. Every node contains 91 network-traffic features.

4. Examples of node-level information include:
   - Duration
   - Packet statistics
   - Byte statistics
   - Traffic rate
   - Loss information
   - TCP/UDP/network protocol indicators
   - Other encoded network-flow characteristics

5. The graph contains 5 nodes because the current
   progressive window size is 5.

6. Global StandardScaler is used during edge construction
   to normalize the feature space.

7. k-NN with k = 2 identifies the two nearest traffic-flow
   nodes for every node.

8. Each k-NN relationship becomes a directed graph edge.

9. Self-loops are removed during edge creation.

10. Duplicate edges are removed during edge creation.

11. The resulting graph is stored in PyTorch Geometric
    format by pyg_dataset.py.

12. Five consecutive graph snapshots are later grouped
    into one temporal sequence by sequence_dataset.py.

13. These temporal graph sequences are provided to the
    HTSTCL-GNN model for spatial and temporal learning.
"""
)

print("\n")
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "Run: python models/pyg_dataset.py"
)

print("=" * 70)