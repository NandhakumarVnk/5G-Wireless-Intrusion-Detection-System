import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print("=" * 60)
print("GRAPH BUILDER MODULE")
print("=" * 60)

# =====================================================
# Configuration
# =====================================================

WINDOW_SIZE = 5

# None = Build all graphs
MAX_GRAPHS = 500

# =====================================================
# Create Output Folder
# =====================================================

output_dir = "outputs/graphs"

os.makedirs(

    output_dir,

    exist_ok=True

)

# =====================================================
# Load CSV Files
# =====================================================

window_file = (

    f"outputs/progressive_windows/window_{WINDOW_SIZE}.csv"

)

edge_file = (

    f"outputs/graph_edges/edges_window_{WINDOW_SIZE}.csv"

)

print("\nLoading Dataset...\n")

window_df = pd.read_csv(

    window_file

)

edge_df = pd.read_csv(

    edge_file

)

print("Window File :", window_file)

print("Edge File   :", edge_file)

print("\nWindow Dataset Shape :", window_df.shape)

print("Edge Dataset Shape   :", edge_df.shape)

# =====================================================
# Determine Total Graphs Automatically
# =====================================================

available_graphs = sorted(

    window_df["Window_ID"].unique()

)

total_graphs = len(

    available_graphs

)

print("\n")
print("=" * 60)
print("GRAPH INFORMATION")
print("=" * 60)

print("Total Graphs Available :", total_graphs)

if MAX_GRAPHS is None:

    graphs_to_build = available_graphs

    print("Graphs To Build        : ALL")

else:

    graphs_to_build = available_graphs[:MAX_GRAPHS]

    print("Graphs To Build        :", len(graphs_to_build))

generated = 0

print("\n")
print("=" * 60)
print("BUILDING NETWORKX GRAPHS")
print("=" * 60)
# =====================================================
# Build NetworkX Graphs
# =====================================================

for graph_id in graphs_to_build:

    # -------------------------------------------------
    # Extract Current Graph
    # -------------------------------------------------

    graph_window = window_df[

        window_df["Window_ID"] == graph_id

    ]

    graph_edges = edge_df[

        edge_df["Window_ID"] == graph_id

    ]

    if len(graph_window) == 0:

        continue

    # -------------------------------------------------
    # Create NetworkX Graph
    # -------------------------------------------------

    G = nx.Graph()

    # -------------------------------------------------
    # Add Nodes
    # -------------------------------------------------

    for node in range(len(graph_window)):

        G.add_node(node)

    # -------------------------------------------------
    # Add Edges
    # -------------------------------------------------

    for _, row in graph_edges.iterrows():

        source = int(row["Source"])

        target = int(row["Target"])

        G.add_edge(

            source,

            target

        )

    # -------------------------------------------------
    # Print Graph Information (First 10 Only)
    # -------------------------------------------------

    if generated < 10:

        print("\n")
        print("=" * 60)
        print(f"GRAPH SNAPSHOT : {graph_id}")
        print("=" * 60)

        print("Window ID          :", graph_id)

        print("Number of Nodes    :", G.number_of_nodes())

        print("Number of Edges    :", G.number_of_edges())

        print("\nNodes")

        print(

            list(G.nodes())

        )

        print("\nEdges")

        print(

            list(G.edges())

        )

        print("\nDegree of Each Node")

        for node, degree in G.degree():

            print(

                f"Node {node} --> Degree : {degree}"

            )

    # -------------------------------------------------
    # Graph Visualization
    # -------------------------------------------------

    plt.figure(

        figsize=(6, 5)

    )

    pos = nx.spring_layout(

        G,

        seed=42

    )

    nx.draw_networkx_nodes(

        G,

        pos,

        node_size=700,

        node_color="skyblue",

        edgecolors="black"

    )

    nx.draw_networkx_edges(

        G,

        pos,

        width=2

    )

    nx.draw_networkx_labels(

        G,

        pos,

        font_size=10,

        font_weight="bold"

    )

    plt.title(

        f"Graph {graph_id}",

        fontsize=13

    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            output_dir,

            f"graph_snapshot_{graph_id}.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    generated += 1

    # -------------------------------------------------
    # Progress
    # -------------------------------------------------

    if generated % 100 == 0:

        print(

            f"{generated} Graph Images Generated..."

        )
        # =====================================================
# Dataset Statistics
# =====================================================

print("\n")
print("=" * 60)
print("GRAPH DATASET STATISTICS")
print("=" * 60)

total_nodes = len(window_df)

total_edges = len(edge_df)

print("Graphs Built           :", generated)

print("Total Nodes            :", total_nodes)

print("Total Edges            :", total_edges)

print("Average Nodes / Graph  :",

      total_nodes / generated)

print("Average Edges / Graph  :",

      total_edges / generated)

# =====================================================
# Verify Generated Images
# =====================================================

print("\n")
print("=" * 60)
print("VERIFY GENERATED IMAGES")
print("=" * 60)

image_files = [

    file

    for file in os.listdir(output_dir)

    if file.endswith(".png")

]

print("Images Generated :", len(image_files))

print("\nFirst 10 Images")

for image in image_files[:10]:

    print(image)

# =====================================================
# Final Summary
# =====================================================

print("\n")
print("=" * 60)
print("GRAPH BUILDER COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Window Size        :", WINDOW_SIZE)

print("Graphs Available   :", total_graphs)

print("Graphs Built       :", generated)

print("Graph Images Saved :", len(image_files))

print("Output Folder      :", output_dir)

print("\nGraph Dataset")

print("----------------------------------------")

print("Nodes :", total_nodes)

print("Edges :", total_edges)

print("Average Nodes / Graph :",

      f"{total_nodes / generated:.2f}")

print("Average Edges / Graph :",

      f"{total_edges / generated:.2f}")

print("\nVerification")

print("----------------------------------------")

if generated == len(image_files):

    print("✔ All graph images generated successfully.")

else:

    print("✘ Some graph images are missing.")

if generated == total_graphs:

    print("✔ All available graphs were processed.")

else:

    print(f"✔ Processed {generated} of {total_graphs} graphs.")

print("✔ Graph visualization completed.")

print("✔ Compatible with pyg_dataset.py")

print("✔ Ready for graph_dataset.pt generation")

print("\n")
print("=" * 60)
print("NEXT STEP")
print("=" * 60)

print("Run:")

print("python models/pyg_dataset.py")

print("=" * 60)