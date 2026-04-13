import math

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class ODCrossAttention(nn.Module):
    """
    基于 OD 图的多头交叉注意力，带边偏置（flow）。
    边方向：src -> dst，其中 src 为目的地，dst 为源（聚合到 dst）。

    设计目标：让不同 head 负责不同模态的交叉注意力（语义分工）。
    - head0: res -> poi (仅用于计算注意力分数)
    - head1: res -> vis (仅用于计算注意力分数)
    - head2: res -> fused(poi+vis+street) (仅用于计算注意力分数)
    - head3: res -> street (仅用于计算注意力分数，街景专用)
    - 若 n_heads > 4：head4... 默认也使用 fused (仅用于计算注意力分数)
    - Value 使用整体特征 h_spatial，而不是子空间特征
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
        assert hidden_dim % n_heads == 0, "hidden_dim 必须能被 n_heads 整除"
        assert n_heads >= 4, "为了实现 head0/1/2/3 的模态分工，n_heads 必须 >= 4"
        self.head_dim = hidden_dim // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(in_q_dim, hidden_dim, bias=False)
        # 每个模态各自映射到 head_dim，仅用于计算注意力分数（Key）
        # 支持消融实验：如果维度为0，则不创建对应的Linear层
        self.w_k_poi = nn.Linear(in_poi_dim, self.head_dim, bias=False) if in_poi_dim > 0 else None
        self.w_k_vis = nn.Linear(in_vis_dim, self.head_dim, bias=False) if in_vis_dim > 0 else None
        self.w_k_street = nn.Linear(
            in_street_dim, self.head_dim, bias=False) if in_street_dim > 0 else None
        self.w_k_fused = nn.Linear(in_fused_dim, self.head_dim, bias=False) if in_fused_dim > 0 else None
        # Value 使用整体特征，投影到 hidden_dim
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
        g: OD 图，方向 src->dst
        q_feats: [N, q_dim]，用于 Query（来自 res 子向量）
        poi_feats: [N, poi_dim]，用于 head0 的 Key（仅用于计算注意力分数）
        vis_feats: [N, vis_dim]，用于 head1 的 Key（仅用于计算注意力分数）
        street_feats: [N, street_dim]，用于 head3 的 Key（仅用于计算注意力分数，街景专用）
        fused_feats: [N, fused_dim]，用于 head2(及其后续 head) 的 Key（仅用于计算注意力分数）
        edge_flow: [E, 1]，流量特征
        h_spatial: [N, hidden_dim]，整体特征，用于 Value（聚合时使用）
        """
        n = q_feats.shape[0]
        device = q_feats.device

        # Query: 使用子空间特征
        q = self.w_q(q_feats).view(n, self.n_heads, self.head_dim)

        # Key: 按 head 分配不同模态的子空间特征（仅用于计算注意力分数）
        k = torch.zeros((n, self.n_heads, self.head_dim),
                        device=device, dtype=q.dtype)

        # 支持消融实验：如果某个模态被禁用（维度为0），使用fused或零向量
        # 先计算所有可用的k值
        k_poi = self.w_k_poi(poi_feats) if self.w_k_poi is not None else None
        k_vis = self.w_k_vis(vis_feats) if self.w_k_vis is not None else None
        k_fused = self.w_k_fused(fused_feats) if self.w_k_fused is not None else None

        # head0: 使用POI（如果可用），否则使用fused或零向量
        if k_poi is not None:
            k[:, 0, :] = k_poi
        elif k_fused is not None:
            k[:, 0, :] = k_fused
        # 否则保持为0（已在初始化时设为0）

        # head1: 使用vis（如果可用），否则使用fused或零向量
        if k_vis is not None:
            k[:, 1, :] = k_vis
        elif k_fused is not None:
            k[:, 1, :] = k_fused
        # 否则保持为0

        # head2: 使用fused（如果可用）
        if k_fused is not None:
            k[:, 2, :] = k_fused

        # head3 使用街景特征
        if self.w_k_street is not None:
            k_street = self.w_k_street(street_feats)  # [N, head_dim]
            k[:, 3, :] = k_street
            # head4 及以后使用 fused（如果fused可用）
            if self.n_heads > 4 and self.w_k_fused is not None:
                k[:, 4:, :] = k_fused.unsqueeze(
                    1).expand(-1, self.n_heads - 4, -1)
        else:
            # 如果没有街景，head3 及以后都使用 fused（如果fused可用）
            if self.n_heads > 3 and self.w_k_fused is not None:
                k[:, 3:, :] = k_fused.unsqueeze(
                    1).expand(-1, self.n_heads - 3, -1)

        # Value: 使用整体特征 h_spatial，投影到 hidden_dim
        v = self.w_v(h_spatial)  # [N, hidden_dim]
        # 将 Value 按 head 分割，每个 head 使用 head_dim 维度
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

        # 聚合：使用整体特征的 Value，按注意力权重聚合
        g.update_all(fn.u_mul_e("v", "a", "m"), fn.sum("m", "h"))
        h = g.ndata["h"].reshape(-1, self.n_heads * self.head_dim)
        q_residual = q.reshape(-1, self.n_heads * self.head_dim)
        h = self.dropout(self.out(h))
        return self.norm(h + q_residual)


class ODCrossAttentionUnified(nn.Module):
    """
    消融变体：去除分模态交互机制（W/o Interaction）。

    与 ODCrossAttention 的差异：
    - 不再为不同 head 指派不同模态 Key 子空间；
    - 只使用“全模态融合特征 fused_feats”作为 Key，且所有 head 共用同一套 Key；
    - Value 仍使用整体特征 h_spatial。
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
        assert hidden_dim % n_heads == 0, "hidden_dim 必须能被 n_heads 整除"
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
