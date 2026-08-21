import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool

from spatial_encoder import SpatialEncoder
from temporal_encoder import TemporalEncoder
from hierarchical_fusion import HierarchicalFusion


class HTSTCL_GNN(nn.Module):

    def __init__(
        self,
        input_dim=91,
        hidden_dim=128,
        embedding_dim=128,
        num_classes=2,
        gru_layers=2,
        dropout=0.3
    ):

        super().__init__()

        # =====================================================
        # Configuration
        # =====================================================

        self.embedding_dim = embedding_dim

        # =====================================================
        # Spatial Encoder
        # =====================================================

        self.spatial_encoder = SpatialEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            heads=4,
            dropout=dropout
        )

        # =====================================================
        # Temporal Encoder
        # =====================================================

        self.temporal_encoder = TemporalEncoder(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_layers=gru_layers,
            dropout=dropout,
            bidirectional=False
        )

        # =====================================================
        # Hierarchical Fusion
        # =====================================================

        self.fusion = HierarchicalFusion(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # =====================================================
        # Projection Head
        # =====================================================

        self.projection_head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                embedding_dim,
                embedding_dim
            )
        )

        # =====================================================
        # Graph / Sequence Classifier
        # =====================================================

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                64
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                num_classes
            )
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, batch_sequences):
        """
        Parameters
        ----------
        batch_sequences : list

        Each element is a dictionary

        {
            "graphs": [Graph1, Graph2, Graph3, Graph4, Graph5],
            "label": 0 or 1
        }

        Returns
        -------
        Dictionary containing logits, embeddings,
        projection vectors and attention weights.
        """

        batch_spatial_sequences = []

        # =====================================================
        # Process every temporal sequence in the batch
        # =====================================================

        for sample in batch_sequences:

            graph_sequence = sample["graphs"]

            sequence_embeddings = []

            # -------------------------------------------------
            # Spatial Encoding for every graph
            # -------------------------------------------------

            for graph in graph_sequence:

                x = graph.x
                edge_index = graph.edge_index

                # ---------------------------------------------
                # Node Embeddings using GAT
                # ---------------------------------------------

                node_embedding = self.spatial_encoder(
                    x,
                    edge_index
                )

                # ---------------------------------------------
                # Graph Embedding using Global Mean Pooling
                # ---------------------------------------------

                batch = torch.zeros(
                    graph.num_nodes,
                    dtype=torch.long,
                    device=x.device
                )

                graph_embedding = global_mean_pool(
                    node_embedding,
                    batch
                )

                sequence_embeddings.append(
                    graph_embedding.squeeze(0)
                )

            # ---------------------------------------------
            # Sequence Tensor
            #
            # Shape:
            # Sequence Length × Embedding
            #
            # Example:
            # 5 × 128
            # ---------------------------------------------

            sequence_embeddings = torch.stack(
                sequence_embeddings,
                dim=0
            )

            batch_spatial_sequences.append(
                sequence_embeddings
            )

        # =====================================================
        # Batch Tensor
        #
        # Shape:
        # Batch × Sequence × Embedding
        #
        # Example:
        # 16 × 5 × 128
        # =====================================================

        temporal_input = torch.stack(
            batch_spatial_sequences,
            dim=0
        )

        # =====================================================
        # Temporal Encoder
        # =====================================================

        temporal_embedding = self.temporal_encoder(
            temporal_input
        )

        # Shape:
        # Batch × Embedding
        #
        # Example:
        # 16 × 128

        # =====================================================
        # Current Spatial Representation
        # (Last graph of each sequence)
        # =====================================================

        current_spatial = temporal_input[:, -1, :]

        # =====================================================
        # Hierarchical Fusion
        # =====================================================

        fused_embedding, spatial_weight, temporal_weight = self.fusion(
            current_spatial,
            temporal_embedding
        )

        # =====================================================
        # Projection Head
        # Used for Contrastive Learning
        # =====================================================

        projection = self.projection_head(
            fused_embedding
        )

        # Shape
        #
        # Batch × Embedding
        #
        # Example
        #
        # 16 × 128

        # =====================================================
        # Sequence Classification
        # =====================================================

        logits = self.classifier(
            fused_embedding
        )

        # Shape
        #
        # Batch × Classes
        #
        # Example
        #
        # 16 × 2

        # =====================================================
        # Return Outputs
        # =====================================================

        return {
            "logits": logits,
            "embedding": fused_embedding,
            "projection": projection,
            "spatial_embedding": current_spatial,
            "temporal_embedding": temporal_embedding,
            "spatial_weight": spatial_weight,
            "temporal_weight": temporal_weight
        }