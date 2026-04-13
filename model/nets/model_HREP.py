from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.layers.projection_head import ProjectionHead


def _dense_mobility_from_od_graph(g_od: dgl.DGLGraph, num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Build dense mobility matrix M [N, N] from OD graph edges.
    Uses g_od.edata['flow_raw'] if present, else 'flow', else 1.
    """
    M = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
    if g_od.num_edges() == 0:
        return M
    src, dst = g_od.edges()
    src = src.to(device)
    dst = dst.to(device)
    if "flow_raw" in g_od.edata:
        w = g_od.edata["flow_raw"].to(device).view(-1).float()
    elif "flow" in g_od.edata:
        w = g_od.edata["flow"].to(device).view(-1).float()
    else:
        w = torch.ones((src.numel(),), device=device, dtype=torch.float32)
    # allow multi-edges: sum
    M.index_put_((src, dst), w, accumulate=True)
    return M


def _cosine_sim_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    x = x / (x.norm(dim=1, keepdim=True) + eps)
    return x @ x.t()


def _topk_edge_index(sim: torch.Tensor, k: int) -> torch.Tensor:
    """
    sim: [N, N] dense similarity (float)
    return edge_index: [2, E] long, undirected, includes self-loops
    """
    n = sim.size(0)
    k = min(int(k), n)
    # include self and top-k neighbors (per column like original code)
    topk = torch.topk(sim, k=k, dim=0, largest=True).indices  # [k, N]
    col = torch.arange(n, device=sim.device).view(1, n).expand(k, n)
    rows = topk.reshape(-1)
    cols = col.reshape(-1)
    # symmetric
    edge_u = torch.cat([rows, cols], dim=0)
    edge_v = torch.cat([cols, rows], dim=0)
    edge_index = torch.stack([edge_u, edge_v], dim=0)
    return edge_index


def _edge_index_to_dgl_graph(edge_index: torch.Tensor, num_nodes: int) -> dgl.DGLGraph:
    u, v = edge_index[0], edge_index[1]
    g = dgl.graph((u, v), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    return g


def _neighbor_lists_from_spatial(g_spatial: dgl.DGLGraph) -> list[list[int]]:
    """
    Build Python neighbor lists from spatial graph (undirected assumed/treated).
    """
    n = g_spatial.num_nodes()
    src, dst = g_spatial.edges()
    src = src.cpu().tolist()
    dst = dst.cpu().tolist()
    neigh = [set() for _ in range(n)]
    for s, d in zip(src, dst):
        neigh[s].add(d)
        neigh[d].add(s)
    return [sorted(list(s)) for s in neigh]


def _sample_pos_neg(neighbor_lists: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(neighbor_lists)
    pos = torch.empty((n,), device=device, dtype=torch.long)
    neg = torch.empty((n,), device=device, dtype=torch.long)
    for i in range(n):
        nbrs = neighbor_lists[i]
        if len(nbrs) > 0:
            j = nbrs[torch.randint(0, len(nbrs), (1,)).item()]
        else:
            j = i
        pos[i] = j
    for i in range(n):
        if len(neighbor_lists[i]) >= n - 1:
            neg[i] = i
            continue
        j = torch.randint(0, n, (1,)).item()
        nbr_set = set(neighbor_lists[i])
        while j == i or j in nbr_set:
            j = torch.randint(0, n, (1,)).item()
        neg[i] = j
    return pos, neg


class RelationGCN(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, gcn_layers: int):
        super().__init__()
        self.gcn_layers = int(gcn_layers)
        self.dropout = float(dropout)
        self.convs = nn.ModuleList(
            [dgl.nn.GraphConv(embedding_size, embedding_size, norm="both", weight=True, bias=True)
             for _ in range(self.gcn_layers)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(embedding_size) for _ in range(self.gcn_layers - 1)])
        self.rel_trans = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for _ in range(self.gcn_layers)])

    def _apply_one(self, g: dgl.DGLGraph, x: torch.Tensor, r: torch.Tensor, i: int, is_training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if i < self.gcn_layers - 1:
            tmp = x
            x = tmp + F.leaky_relu(self.bns[i](self.convs[i](g, x * r)))
            r = self.rel_trans[i](r)
            if is_training:
                x = F.dropout(x, p=self.dropout, training=True)
            return x, r
        x = self.convs[i](g, x * r)
        r = self.rel_trans[i](r)
        return x, r

    def forward(
        self,
        features: torch.Tensor,
        rel_emb: list[torch.Tensor],
        graphs: list[dgl.DGLGraph],
        is_training: bool = True,
    ):
        poi_r, s_r, d_r, n_r = rel_emb
        g_poi, g_s, g_d, g_n = graphs

        n_emb = features
        poi_emb = features
        s_emb = features
        d_emb = features

        for i in range(self.gcn_layers):
            n_emb, n_r = self._apply_one(g_n, n_emb, n_r, i, is_training)
            poi_emb, poi_r = self._apply_one(g_poi, poi_emb, poi_r, i, is_training)
            s_emb, s_r = self._apply_one(g_s, s_emb, s_r, i, is_training)
            d_emb, d_r = self._apply_one(g_d, d_emb, d_r, i, is_training)

        return n_emb, poi_emb, s_emb, d_emb, n_r, poi_r, s_r, d_r


class CrossLayer(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int = 4):
        super().__init__()
        self.alpha_n = nn.Parameter(torch.tensor(0.95))
        self.alpha_poi = nn.Parameter(torch.tensor(0.95))
        self.alpha_d = nn.Parameter(torch.tensor(0.95))
        self.alpha_s = nn.Parameter(torch.tensor(0.95))
        self.attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=False)

    def forward(self, n_emb: torch.Tensor, poi_emb: torch.Tensor, s_emb: torch.Tensor, d_emb: torch.Tensor):
        stk = torch.stack((n_emb, poi_emb, d_emb, s_emb), dim=0)  # [4, N, D]
        fusion, _ = self.attn(stk, stk, stk, need_weights=False)
        n_f = fusion[0] * self.alpha_n + (1 - self.alpha_n) * n_emb
        poi_f = fusion[1] * self.alpha_poi + (1 - self.alpha_poi) * poi_emb
        d_f = fusion[2] * self.alpha_d + (1 - self.alpha_d) * d_emb
        s_f = fusion[3] * self.alpha_s + (1 - self.alpha_s) * s_emb
        return n_f, poi_f, s_f, d_f


class AttentionFusionLayer(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(embedding_size))
        self.fusion_lin = nn.Linear(embedding_size, embedding_size)

    def forward(self, n_f: torch.Tensor, poi_f: torch.Tensor, s_f: torch.Tensor, d_f: torch.Tensor):
        def w(x: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.sum(F.leaky_relu(self.fusion_lin(x)) * self.q, dim=1))

        w_stk = torch.stack((w(n_f), w(poi_f), w(s_f), w(d_f)), dim=0)
        w_norm = torch.softmax(w_stk, dim=0)
        region_feature = w_norm[0] * n_f + w_norm[1] * poi_f + w_norm[2] * s_f + w_norm[3] * d_f
        return region_feature


class HRE(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, gcn_layers: int):
        super().__init__()
        self.relation_gcns = RelationGCN(embedding_size, dropout, gcn_layers)
        self.cross_layer = CrossLayer(embedding_size)
        self.fusion_layer = AttentionFusionLayer(embedding_size)

    def forward(self, features: torch.Tensor, rel_emb: list[torch.Tensor], graphs: list[dgl.DGLGraph], is_training: bool = True):
        n_emb, poi_emb, s_emb, d_emb, n_r, poi_r, s_r, d_r = self.relation_gcns(features, rel_emb, graphs, is_training)
        n_f, poi_f, s_f, d_f = self.cross_layer(n_emb, poi_emb, s_emb, d_emb)
        region_feature = self.fusion_layer(n_f, poi_f, s_f, d_f)
        n_f = region_feature * n_r
        poi_f = region_feature * poi_r
        s_f = region_feature * s_r
        d_f = region_feature * d_r
        return region_feature, n_f, poi_f, s_f, d_f


class UrbanModelHREP(nn.Module):
    """
    HREP baseline adapter for this codebase.

    Inputs (from UrbanWeekdayDataset batch):
    - g_spatial: DGL spatial adjacency graph
    - g_od: DGL OD graph with edge flow
    - feat: node features [N, poi+res+vis+street]

    Builds HREP required relation graphs on-the-fly:
    - neighbor graph: from g_spatial
    - mobility matrix: from g_od
    - poi similarity: cosine similarity on poi features
    - source/destination adjacency: cosine similarity of mobility row/col patterns
    """

    def __init__(
        self,
        dims: Dict[str, int],
        embedding_size: int = 144,
        gcn_layers: int = 3,
        dropout: float = 0.1,
        importance_k: int = 10,
        tau: float = 0.1,  # unused, kept for config consistency
        proj_dim: int = 128,
    ):
        super().__init__()
        self.dims = dims
        self.embedding_size = int(embedding_size)
        self.gcn_layers = int(gcn_layers)
        self.dropout = float(dropout)
        self.importance_k = int(importance_k)

        self.poi_dim = int(dims["poi"])

        self.hre = HRE(self.embedding_size, self.dropout, self.gcn_layers)
        self.proj = ProjectionHead(self.embedding_size, self.embedding_size, proj_dim)

        # relation embeddings
        self.poi_r = nn.Parameter(torch.randn(self.embedding_size))
        self.s_r = nn.Parameter(torch.randn(self.embedding_size))
        self.d_r = nn.Parameter(torch.randn(self.embedding_size))
        self.n_r = nn.Parameter(torch.randn(self.embedding_size))

        self.triplet = nn.TripletMarginLoss()
        self.mse = nn.MSELoss()

        # cache graphs built from current batch (single-graph dataset)
        self._cache_num_nodes: int | None = None
        self._cache_graphs: list[dgl.DGLGraph] | None = None
        self._cache_neighbor_lists: list[list[int]] | None = None
        self._cache_poi_sim: torch.Tensor | None = None
        self._cache_mobility: torch.Tensor | None = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        h_spatial, h_od, z, loss_terms = self._encode_and_loss(batch)
        info = {"loss": loss_terms["loss"].item(), **{k: v.item() for k, v in loss_terms.items()}, "z": z, "h_spatial": h_spatial, "h_od": h_od}
        return loss_terms["loss"], info

    def encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_spatial, h_od, z, _ = self._encode_and_loss(batch, compute_loss=False)
        return h_spatial, h_od, z

    def _ensure_cached_structures(self, g_spatial: dgl.DGLGraph, g_od: dgl.DGLGraph, feat: torch.Tensor):
        n = g_spatial.num_nodes()
        device = feat.device
        if self._cache_num_nodes == n and self._cache_graphs is not None and self._cache_neighbor_lists is not None:
            # device migration if needed
            self._cache_graphs = [gg.to(device) for gg in self._cache_graphs]
            if self._cache_poi_sim is not None:
                self._cache_poi_sim = self._cache_poi_sim.to(device)
            if self._cache_mobility is not None:
                self._cache_mobility = self._cache_mobility.to(device)
            return

        neighbor_lists = _neighbor_lists_from_spatial(g_spatial)
        mobility = _dense_mobility_from_od_graph(g_od, n, device=device)

        poi_feat = feat[:, : self.poi_dim]
        poi_sim = _cosine_sim_matrix(poi_feat)

        # source/destination adjacency: similarity of mobility patterns
        s_adj = _cosine_sim_matrix(mobility + 1e-6)  # rows
        d_adj = _cosine_sim_matrix(mobility.t() + 1e-6)  # cols

        g_poi = _edge_index_to_dgl_graph(_topk_edge_index(poi_sim, self.importance_k), n)
        g_s = _edge_index_to_dgl_graph(_topk_edge_index(s_adj, self.importance_k), n)
        g_d = _edge_index_to_dgl_graph(_topk_edge_index(d_adj, self.importance_k), n)

        # neighbor graph: full adjacency from spatial (undirected) + self-loop
        src, dst = g_spatial.edges()
        g_n = dgl.graph((src.to(device), dst.to(device)), num_nodes=n)
        g_n = dgl.to_simple(dgl.add_reverse_edges(g_n))
        g_n = dgl.add_self_loop(g_n)

        self._cache_num_nodes = n
        self._cache_graphs = [g_poi.to(device), g_s.to(device), g_d.to(device), g_n.to(device)]
        self._cache_neighbor_lists = neighbor_lists
        self._cache_poi_sim = poi_sim
        self._cache_mobility = mobility

    def _mob_loss(self, s_emb: torch.Tensor, d_emb: torch.Tensor, mob: torch.Tensor) -> torch.Tensor:
        inner = torch.mm(s_emb, d_emb.t())
        ps_hat = F.softmax(inner, dim=-1)
        inner2 = torch.mm(d_emb, s_emb.t())
        pd_hat = F.softmax(inner2, dim=-1)
        mob = mob / (mob.mean() + 1e-8)
        # 对 N×N 项取均值而非求和，否则 loss 随节点数平方增长（易达 1e8 量级）
        ce = -mob * torch.log(ps_hat + 1e-8) - mob * torch.log(pd_hat + 1e-8)
        return ce.mean()

    def _encode_and_loss(self, batch: Dict[str, torch.Tensor], compute_loss: bool = True):
        g_spatial = batch["g_spatial"]
        g_od = batch["g_od"]
        feat = batch["feat"]
        device = feat.device

        self._ensure_cached_structures(g_spatial, g_od, feat)
        assert self._cache_graphs is not None and self._cache_neighbor_lists is not None
        graphs = self._cache_graphs
        neighbor_lists = self._cache_neighbor_lists
        mobility = self._cache_mobility
        poi_sim = self._cache_poi_sim

        # feature projection to embedding_size
        # Use a simple linear projection initialized lazily to match input dim
        if not hasattr(self, "feat_proj"):
            self.feat_proj = nn.Linear(feat.size(1), self.embedding_size).to(device)  # type: ignore[attr-defined]
        x = self.feat_proj(feat)  # [N, embedding_size]

        rel_emb = [self.poi_r, self.s_r, self.d_r, self.n_r]
        region_emb, n_emb, poi_emb, s_emb, d_emb = self.hre(x, rel_emb, graphs, is_training=self.training)

        z = F.normalize(self.proj(region_emb), dim=-1)
        h_spatial = region_emb
        h_od = torch.zeros_like(region_emb)

        if not compute_loss:
            return h_spatial, h_od, z, {}

        pos_idx, neg_idx = _sample_pos_neg(neighbor_lists, device=device)
        geo_loss = self.triplet(n_emb, n_emb[pos_idx], n_emb[neg_idx])
        m_loss = self._mob_loss(s_emb, d_emb, mobility)
        # poi_sim 为余弦相似度 [-1,1]；点积矩阵未 L2 归一化时量级差几个数量级，MSE 会失真
        poi_loss = self.mse(_cosine_sim_matrix(poi_emb), poi_sim)
        loss = poi_loss + m_loss + geo_loss
        return h_spatial, h_od, z, {"loss": loss, "poi_loss": poi_loss, "mob_loss": m_loss, "geo_loss": geo_loss}

