import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class SpatialEncoder(nn.Module):

    def __init__(
        self,
        input_dim=91,
        hidden_dim=128,
        output_dim=128,
        heads=4,
        dropout=0.3
    ):

        super().__init__()

        # First Graph Attention Layer
        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        # Second Graph Attention Layer
        self.gat2 = GATConv(
            hidden_dim * heads,
            output_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x, edge_index):

        # Layer 1
        x = self.gat1(x, edge_index)

        x = F.elu(x)

        x = self.dropout(x)

        # Layer 2
        x = self.gat2(x, edge_index)

        x = self.batch_norm(x)

        x = F.elu(x)

        return x