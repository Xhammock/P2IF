import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn


class SpatialSAGE(nn.Module):
    """
    简单的 GraphSAGE（mean 聚合），用于空间邻接图。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(
                dglnn.SAGEConv(
                    in_feats=dims[i],
                    out_feats=dims[i + 1],
                    aggregator_type="mean",
                    feat_drop=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph, feats: torch.Tensor) -> torch.Tensor:
        h = feats
        for conv in self.layers:
            h = conv(g, h)
            h = self.act(h)
            h = self.dropout(h)
        return h

