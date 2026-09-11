
import torch
import torch.nn as nn

from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from .spatial_encoder import SpatialEncoder
from .temporal_encoder import TemporalEncoder
from .hierarchical_fusion import HierarchicalFusion


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

        self.embedding_dim = embedding_dim

        # =====================================================
        # SPATIAL ENCODER
        # =====================================================

        self.spatial_encoder = SpatialEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            heads=4,
            dropout=dropout
        )

        # =====================================================
        # TEMPORAL ENCODER
        # =====================================================

        self.temporal_encoder = TemporalEncoder(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_layers=gru_layers,
            dropout=dropout,
            bidirectional=False
        )

        # =====================================================
        # HIERARCHICAL FUSION
        # =====================================================

        self.fusion = HierarchicalFusion(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # =====================================================
        # PROJECTION HEAD
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
        # CLASSIFIER
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
    # FORWARD
    # =====================================================

    def forward(self, batch_sequences):

        """
        batch_sequences:

        List of sequences

        Shape concept:

        Batch
            ↓
        Sequence Length = 5
            ↓
        Graphs

        Example:

        [
            [G1, G2, G3, G4, G5],
            [G1, G2, G3, G4, G5],
            ...
        ]
        """

        # =====================================================
        # GET MODEL DEVICE
        # =====================================================

        device = next(
            self.parameters()
        ).device

        batch_size = len(
            batch_sequences
        )

        sequence_length = len(
            batch_sequences[0]
        )

        temporal_embeddings = []

        # =====================================================
        # PROCESS EACH TEMPORAL STEP
        #
        # Instead of:
        #
        # for every sample
        #     for every graph
        #         run GAT
        #
        # We do:
        #
        # Collect all graphs at time step t
        # Batch them together
        # Run GAT only once
        #
        # =====================================================

        for time_step in range(
            sequence_length
        ):

            # -------------------------------------------------
            # COLLECT GRAPHS FROM ALL SAMPLES
            # -------------------------------------------------

            graphs_at_time_step = [

                batch_sequences[
                    batch_index
                ][
                    time_step
                ]

                for batch_index in range(
                    batch_size
                )
            ]

            # -------------------------------------------------
            # CREATE PYG BATCH
            # -------------------------------------------------

            batched_graph = Batch.from_data_list(
                graphs_at_time_step
            )

            # -------------------------------------------------
            # IMPORTANT FIX
            #
            # Move entire PyG Batch to same device as model
            # -------------------------------------------------

            batched_graph = batched_graph.to(
                device
            )

            # -------------------------------------------------
            # SPATIAL ENCODING
            #
            # All graphs processed together
            # -------------------------------------------------

            node_embeddings = self.spatial_encoder(

                batched_graph.x,

                batched_graph.edge_index

            )

            # -------------------------------------------------
            # GRAPH LEVEL POOLING
            #
            # Output:
            #
            # Batch × Embedding
            #
            # Example:
            #
            # 32 × 128
            # -------------------------------------------------

            graph_embeddings = global_mean_pool(

                node_embeddings,

                batched_graph.batch

            )

            temporal_embeddings.append(
                graph_embeddings
            )

        # =====================================================
        # STACK TEMPORAL REPRESENTATIONS
        #
        # Current:
        #
        # [
        #   Batch × Embedding,
        #   Batch × Embedding,
        #   ...
        # ]
        #
        # Convert to:
        #
        # Batch × Sequence × Embedding
        # =====================================================

        temporal_input = torch.stack(

            temporal_embeddings,

            dim=1

        )

        # =====================================================
        # TEMPORAL ENCODER
        # =====================================================

        temporal_embedding = self.temporal_encoder(
            temporal_input
        )

        # =====================================================
        # CURRENT SPATIAL REPRESENTATION
        #
        # Last graph in temporal sequence
        # =====================================================

        current_spatial = temporal_input[
            :,
            -1,
            :
        ]

        # =====================================================
        # HIERARCHICAL FUSION
        # =====================================================

        (

            fused_embedding,

            spatial_weight,

            temporal_weight

        ) = self.fusion(

            current_spatial,

            temporal_embedding

        )

        # =====================================================
        # PROJECTION
        # =====================================================

        projection = self.projection_head(
            fused_embedding
        )

        # =====================================================
        # CLASSIFICATION
        # =====================================================

        logits = self.classifier(
            fused_embedding
        )

        # =====================================================
        # RETURN
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