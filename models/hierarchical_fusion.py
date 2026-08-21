import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalFusion(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
        dropout=0.3
    ):

        super().__init__()

        # =====================================================
        # Spatial Attention
        # =====================================================

        self.spatial_attention = nn.Linear(
            embedding_dim,
            1
        )

        # =====================================================
        # Temporal Attention
        # =====================================================

        self.temporal_attention = nn.Linear(
            embedding_dim,
            1
        )

        # =====================================================
        # Feature Refinement
        # =====================================================

        self.fusion_layer = nn.Sequential(

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
    # Forward
    # =====================================================

    def forward(

        self,

        spatial_embedding,

        temporal_embedding

    ):

        """
        Input

        spatial_embedding

            Shape

            Batch × Embedding

            Example

            32 × 128

        temporal_embedding

            Shape

            Batch × Embedding

            Example

            32 × 128
        """

        # ---------------------------------------------
        # Attention Scores
        # ---------------------------------------------

        spatial_score = self.spatial_attention(
            spatial_embedding
        )

        temporal_score = self.temporal_attention(
            temporal_embedding
        )

        attention_scores = torch.cat(

            [

                spatial_score,

                temporal_score

            ],

            dim=1

        )

        attention_weights = F.softmax(

            attention_scores,

            dim=1

        )

        spatial_weight = attention_weights[:, 0].unsqueeze(1)

        temporal_weight = attention_weights[:, 1].unsqueeze(1)

        # ---------------------------------------------
        # Weighted Fusion
        # ---------------------------------------------

        fused_embedding = (

            spatial_weight * spatial_embedding +

            temporal_weight * temporal_embedding

        )

        # ---------------------------------------------
        # Feature Refinement
        # ---------------------------------------------

        fused_embedding = self.fusion_layer(

            fused_embedding

        )

        return (

            fused_embedding,

            spatial_weight,

            temporal_weight

        )