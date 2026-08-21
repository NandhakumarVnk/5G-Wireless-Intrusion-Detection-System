import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # -------------------------------------------------
        # Bidirectional GRU
        # -------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # -------------------------------------------------
        # Projection Layer
        # -------------------------------------------------

        if bidirectional:
            self.projection = nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
        else:
            self.projection = nn.Linear(
                hidden_dim,
                hidden_dim
            )

        self.dropout = nn.Dropout(dropout)

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    def forward(self, sequence_embeddings):

        """
        Input
        -----
        sequence_embeddings

        Shape:
        Batch × Sequence Length × Feature

        Example:

        32 × 5 × 128

        where

        5 =
        Window1
        Window2
        Window3
        Window4
        Window5
        """

        output, hidden = self.gru(sequence_embeddings)

        # ---------------------------------------------
        # Last Time Step
        # ---------------------------------------------

        temporal_embedding = output[:, -1, :]

        temporal_embedding = self.projection(
            temporal_embedding
        )

        temporal_embedding = self.dropout(
            temporal_embedding
        )

        return temporal_embedding