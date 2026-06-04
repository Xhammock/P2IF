import math

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class ODCrossAttention(nn.Module):
    """
    Multi-head cross-attention on the OD graph with edge bias (flow).
    Edge direction: src -> dst, where src is the destination and dst is the source
    (messages aggregate at dst).

    Design: assign different heads to cross-attention over different modalities.
    - head0: res -> poi (attention scores only)
    - head1: res -> vis (attention scores only)
    - head2: res -> fused(poi+vis+street) (attention scores only)
    - head3: res -> street (attention scores only; street-view dedicated)
    - if n_heads > 4: head4+ default to fused (attention scores only)
    - Value uses full representation h_spatial, not subspace features
    """

    def __init__(
        self,
        in_q_dim: int,
        in_poi_dim: int,
        in_vis_dim: int,
        in_street_dim: int,
        in_fused_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        assert n_heads >= 4, (
            "n_heads must be >= 4 for head0/1/2/3 modality specialization"
        )
        self.head_dim = hidden_dim // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(in_q_dim, hidden_dim, bias=False)
        # Per-modality projection to head_dim for attention scores (Key) only
        # Ablation: skip Linear layers when input dimension is 0
        self.w_k_poi = nn.Linear(in_poi_dim, self.head_dim, bias=False) if in_poi_dim > 0 else None
        self.w_k_vis = nn.Linear(in_vis_dim, self.head_dim, bias=False) if in_vis_dim > 0 else None
        self.w_k_street = nn.Linear(
            in_street_dim, self.head_dim, bias=False) if in_street_dim > 0 else None
        self.w_k_fused = nn.Linear(in_fused_dim, self.head_dim, bias=False) if in_fused_dim > 0 else None
        # Value: full features projected to hidden_dim
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_heads),
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        q_feats: torch.Tensor,
        poi_feats: torch.Tensor,
        vis_feats: torch.Tensor,
        street_feats: torch.Tensor,
        fused_feats: torch.Tensor,
        edge_flow: torch.Tensor,
        h_spatial: torch.Tensor,
    ):
        """
        g: OD graph, direction src->dst
        q_feats: [N, q_dim], Query (from res sub-vector)
        poi_feats: [N, poi_dim], Key for head0 (attention scores only)
        vis_feats: [N, vis_dim], Key for head1 (attention scores only)
        street_feats: [N, street_dim], Key for head3 (attention scores only; street-view)
        fused_feats: [N, fused_dim], Key for head2 and later heads (attention scores only)
        edge_flow: [E, 1], flow features
        h_spatial: [N, hidden_dim], full features for Value (aggregation)
        """
        n = q_feats.shape[0]
        device = q_feats.device

        # Query: subspace features
        q = self.w_q(q_feats).view(n, self.n_heads, self.head_dim)

        # Key: per-head modality subspace (attention scores only)
        k = torch.zeros((n, self.n_heads, self.head_dim),
                        device=device, dtype=q.dtype)

        # Ablation: if a modality is disabled (dim 0), use fused or zeros
        # Precompute all available k projections
        k_poi = self.w_k_poi(poi_feats) if self.w_k_poi is not None else None
        k_vis = self.w_k_vis(vis_feats) if self.w_k_vis is not None else None
        k_fused = self.w_k_fused(fused_feats) if self.w_k_fused is not None else None

        # head0: POI if available, else fused or zeros
        if k_poi is not None:
            k[:, 0, :] = k_poi
        elif k_fused is not None:
            k[:, 0, :] = k_fused
        # else remain zero (initialized to 0)

        # head1: vis if available, else fused or zeros
        if k_vis is not None:
            k[:, 1, :] = k_vis
        elif k_fused is not None:
            k[:, 1, :] = k_fused
        # else remain zero

        # head2: fused if available
        if k_fused is not None:
            k[:, 2, :] = k_fused

        # head3: street-view features
        if self.w_k_street is not None:
            k_street = self.w_k_street(street_feats)  # [N, head_dim]
            k[:, 3, :] = k_street
            # head4+: fused if available
            if self.n_heads > 4 and self.w_k_fused is not None:
                k[:, 4:, :] = k_fused.unsqueeze(
                    1).expand(-1, self.n_heads - 4, -1)
        else:
            # No street view: head3+ use fused if available
            if self.n_heads > 3 and self.w_k_fused is not None:
                k[:, 3:, :] = k_fused.unsqueeze(
                    1).expand(-1, self.n_heads - 3, -1)

        # Value: full h_spatial projected to hidden_dim
        v = self.w_v(h_spatial)  # [N, hidden_dim]
        # Split Value per head
        v = v.view(n, self.n_heads, self.head_dim)  # [N, n_heads, head_dim]

        bias = self.edge_mlp(edge_flow).unsqueeze(-1)  # [E, heads, 1]

        g = g.local_var()
        g.ndata["q"] = q
        g.ndata["k"] = k
        g.ndata["v"] = v
        g.edata["bias"] = bias

        # score: (K_src * Q_dst).sum(-1) + bias
        def compute_score(edges):
            score = (edges.src["k"] * edges.dst["q"]).sum(-1, keepdim=True)
            score = score / math.sqrt(self.head_dim) + edges.data["bias"]
            return {"score": score}

        g.apply_edges(compute_score)
        # softmax over incoming edges of dst
        g.edata["a"] = edge_softmax(g, g.edata["score"])

        # Aggregate: Value from full features, weighted by attention
        g.update_all(fn.u_mul_e("v", "a", "m"), fn.sum("m", "h"))
        h = g.ndata["h"].reshape(-1, self.n_heads * self.head_dim)
        q_residual = q.reshape(-1, self.n_heads * self.head_dim)
        h = self.dropout(self.out(h))
        return self.norm(h + q_residual)


class ODCrossAttentionUnified(nn.Module):
    """
    Ablation variant: remove per-modality interaction (W/o Interaction).

    Differences from ODCrossAttention:
    - No per-head modality Key subspaces;
    - Key uses only fused multi-modal features; all heads share the same Key;
    - Value still uses full h_spatial.
    """

    def __init__(
        self,
        in_q_dim: int,
        in_fused_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.head_dim = hidden_dim // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(in_q_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(in_fused_dim, self.head_dim, bias=False) if in_fused_dim > 0 else None
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_heads),
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        q_feats: torch.Tensor,
        fused_feats: torch.Tensor,
        edge_flow: torch.Tensor,
        h_spatial: torch.Tensor,
    ):
        n = q_feats.shape[0]
        device = q_feats.device

        q = self.w_q(q_feats).view(n, self.n_heads, self.head_dim)

        k = torch.zeros((n, self.n_heads, self.head_dim), device=device, dtype=q.dtype)
        if self.w_k is not None:
            k_shared = self.w_k(fused_feats)  # [N, head_dim]
            k = k_shared.unsqueeze(1).expand(-1, self.n_heads, -1).contiguous()

        v = self.w_v(h_spatial).view(n, self.n_heads, self.head_dim)
        bias = self.edge_mlp(edge_flow).unsqueeze(-1)  # [E, heads, 1]

        g = g.local_var()
        g.ndata["q"] = q
        g.ndata["k"] = k
        g.ndata["v"] = v
        g.edata["bias"] = bias

        def compute_score(edges):
            score = (edges.src["k"] * edges.dst["q"]).sum(-1, keepdim=True)
            score = score / math.sqrt(self.head_dim) + edges.data["bias"]
            return {"score": score}

        g.apply_edges(compute_score)
        g.edata["a"] = edge_softmax(g, g.edata["score"])

        g.update_all(fn.u_mul_e("v", "a", "m"), fn.sum("m", "h"))
        h = g.ndata["h"].reshape(-1, self.n_heads * self.head_dim)
        q_residual = q.reshape(-1, self.n_heads * self.head_dim)
        h = self.dropout(self.out(h))
        return self.norm(h + q_residual)
