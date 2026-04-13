from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.layers.spatial_sage import SpatialSAGE
from model.layers.od_cross_attention import ODCrossAttentionUnified
from model.layers.projection_head import ProjectionHead


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        hidden = dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class UrbanModelAugWithoutInteraction(nn.Module):
    """
    消融实验变体：W/o Interaction（去除分模态交互机制）。

    - 空间层：GraphSAGE（与主模型一致）
    - 交互层：将所有模态映射到同一融合空间（fused），使用 ODCrossAttentionUnified 做一次统一注意力融合
    - 训练：保留视图增强与 Spatial-aware NT-Xent（与主模型一致）
    """

    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 256,
        sage_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        proj_dim: int = 128,
        loss_weight: Dict[str, float] | None = None,
        feat_drop_ratio: float = 0.1,
        edge_drop_ratio: float = 0.075,
        noise_std: float = 0.01,
        tau: float = 0.1,
        od_mask_topk: int = 200,
        use_poi: bool = True,
        use_vis: bool = True,
        use_street: bool = True,
    ):
        super().__init__()
        self.dims = dims

        self.use_poi = use_poi
        self.use_vis = use_vis
        self.use_street = use_street

        actual_poi_dim = dims["poi"] if use_poi else 0
        actual_vis_dim = dims["vis"] if use_vis else 0
        actual_street_dim = dims.get("street", 0) if use_street else 0
        in_dim = actual_poi_dim + dims["res"] + actual_vis_dim + actual_street_dim

        self.original_poi_dim = dims["poi"]
        self.original_vis_dim = dims["vis"]
        self.original_street_dim = dims.get("street", 0)
        self.res_dim = dims["res"]

        self.tau = tau
        self.feat_drop_ratio = feat_drop_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.noise_std = noise_std
        self.od_mask_topk = int(od_mask_topk)

        self._spatial_allow_mask: torch.Tensor | None = None
        self._spatial_allow_mask_num_nodes: int | None = None
        self._spatial_allow_mask_num_edges: int | None = None

        self._od_allow_mask: torch.Tensor | None = None
        self._od_allow_mask_num_nodes: int | None = None
        self._od_allow_mask_num_edges: int | None = None

        self.spatial = SpatialSAGE(in_dim, hidden_dim, num_layers=sage_layers, dropout=dropout)

        # Query 仍使用 res 子空间（与主模型一致）
        self.q_dim = dims["res"]
        self.poi_dim = actual_poi_dim
        self.vis_dim = actual_vis_dim
        self.street_dim = actual_street_dim
        self.fused_dim = self.poi_dim + self.vis_dim + self.street_dim

        self.attn = ODCrossAttentionUnified(
            in_q_dim=self.q_dim,
            in_fused_dim=self.fused_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.ffn_spatial = FeedForward(hidden_dim, dropout)
        self.ffn_od = FeedForward(hidden_dim, dropout)
        self.proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)

        self.q_proj = nn.Linear(hidden_dim, self.q_dim)
        self.poi_proj = nn.Linear(hidden_dim, self.poi_dim) if self.poi_dim > 0 else None
        self.vis_proj = nn.Linear(hidden_dim, self.vis_dim) if self.vis_dim > 0 else None
        self.street_proj = nn.Linear(hidden_dim, self.street_dim) if self.street_dim > 0 else None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        view1_spatial, view1_od, view1_feat = self._create_view1(batch["g_spatial"], batch["g_od"], batch["feat"])
        view2_spatial, view2_od, view2_feat = self._create_view2(batch["g_spatial"], batch["g_od"], batch["feat"])

        batch_v1 = {"g_spatial": view1_spatial, "g_od": view1_od, "feat": view1_feat}
        batch_v2 = {"g_spatial": view2_spatial, "g_od": view2_od, "feat": view2_feat}
        h_spatial_v1, h_od_v1, z_v1 = self._encode(batch_v1)
        h_spatial_v2, h_od_v2, z_v2 = self._encode(batch_v2)

        self._ensure_spatial_allow_mask(batch["g_spatial"], device=z_v1.device)
        self._ensure_od_allow_mask(batch["g_od"], device=z_v1.device)

        loss, info = self._contrastive_loss(z_v1, z_v2)
        info.update({"h_spatial": h_spatial_v1, "h_od": h_od_v1, "z": z_v1})
        return loss, info

    def encode(self, batch: Dict[str, torch.Tensor]):
        return self._encode(batch)

    def _extract_features(self, feats: torch.Tensor) -> torch.Tensor:
        parts = []
        start_idx = 0

        if self.use_poi:
            parts.append(feats[:, start_idx:start_idx + self.original_poi_dim])
        start_idx += self.original_poi_dim

        parts.append(feats[:, start_idx:start_idx + self.res_dim])
        start_idx += self.res_dim

        if self.use_vis:
            parts.append(feats[:, start_idx:start_idx + self.original_vis_dim])
        start_idx += self.original_vis_dim

        if self.use_street and self.original_street_dim > 0:
            parts.append(feats[:, start_idx:start_idx + self.original_street_dim])
        if self.original_street_dim > 0:
            start_idx += self.original_street_dim

        return torch.cat(parts, dim=-1) if len(parts) > 0 else feats

    def _encode(self, batch: Dict[str, torch.Tensor]):
        g_spatial = batch["g_spatial"]
        g_od = batch["g_od"]
        feats = self._extract_features(batch["feat"])

        h_spatial = self.spatial(g_spatial, feats)
        h_spatial = self.ffn_spatial(h_spatial)

        q_in = self.q_proj(h_spatial)

        fused_parts = []
        if self.poi_proj is not None:
            fused_parts.append(self.poi_proj(h_spatial))
        if self.vis_proj is not None:
            fused_parts.append(self.vis_proj(h_spatial))
        if self.street_proj is not None:
            fused_parts.append(self.street_proj(h_spatial))
        fused_in = torch.cat(fused_parts, dim=-1) if len(fused_parts) > 0 else torch.zeros(
            (h_spatial.shape[0], 0), device=h_spatial.device, dtype=h_spatial.dtype
        )

        h_od = self.attn(g_od, q_in, fused_in, g_od.edata["flow"], h_spatial)
        h_od = self.ffn_od(h_od)

        h = h_spatial + h_od
        z = F.normalize(self.proj(h), dim=-1)
        return h_spatial, h_od, z

    def _create_view1(self, g_spatial, g_od, feats: torch.Tensor):
        if self.training and self.feat_drop_ratio > 0:
            feat_mask = torch.rand(feats.shape, device=feats.device) > self.feat_drop_ratio
            view1_feat = feats * feat_mask.float()
        else:
            view1_feat = feats

        if self.training and self.noise_std > 0:
            view1_feat = view1_feat + torch.randn_like(view1_feat) * self.noise_std

        view1_spatial = g_spatial.clone()
        view1_spatial.ndata["feat"] = view1_feat.clone()
        view1_od = g_od.clone()
        view1_od.ndata["feat"] = view1_feat.clone()
        return view1_spatial, view1_od, view1_feat

    def _create_view2(self, g_spatial, g_od, feats: torch.Tensor):
        view2_feat = feats
        if self.training and self.noise_std > 0:
            view2_feat = view2_feat + torch.randn_like(view2_feat) * self.noise_std

        view2_spatial = self._drop_edges(g_spatial, self.edge_drop_ratio)
        view2_spatial.ndata["feat"] = view2_feat.clone()
        view2_od = self._drop_edges(g_od, self.edge_drop_ratio)
        view2_od.ndata["feat"] = view2_feat.clone()
        return view2_spatial, view2_od, view2_feat

    def _drop_edges(self, g, drop_ratio: float):
        if not self.training or drop_ratio <= 0:
            return g.clone()
        num_edges = g.num_edges()
        if num_edges == 0:
            return g.clone()
        num_drop = int(num_edges * drop_ratio)
        if num_drop == 0:
            return g.clone()

        device = g.ndata["feat"].device
        eids = torch.randperm(num_edges, device=device)
        keep_eids = eids[num_drop:].sort()[0]

        src, dst = g.edges()
        new_g = dgl.graph((src[keep_eids], dst[keep_eids]), num_nodes=g.num_nodes())
        new_g.ndata["feat"] = g.ndata["feat"].clone()
        if len(g.edata) > 0:
            for key in g.edata:
                new_g.edata[key] = g.edata[key][keep_eids].clone()
        return new_g

    def _ensure_spatial_allow_mask(self, g_spatial, device: torch.device) -> None:
        num_nodes = g_spatial.num_nodes()
        num_edges = g_spatial.num_edges()
        if (
            self._spatial_allow_mask is not None
            and self._spatial_allow_mask.device == device
            and self._spatial_allow_mask_num_nodes == num_nodes
            and self._spatial_allow_mask_num_edges == num_edges
        ):
            return

        adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
        src, dst = g_spatial.edges()
        src = src.to(device)
        dst = dst.to(device)
        adj[src, dst] = True
        adj[dst, src] = True
        adj.fill_diagonal_(True)

        allow_mask = ~adj
        allow_mask.fill_diagonal_(True)

        self._spatial_allow_mask = allow_mask
        self._spatial_allow_mask_num_nodes = num_nodes
        self._spatial_allow_mask_num_edges = num_edges

    def _ensure_od_allow_mask(self, g_od, device: torch.device) -> None:
        num_nodes = g_od.num_nodes()
        num_edges = g_od.num_edges()

        if self.od_mask_topk <= 0 and self.od_mask_topk != -1:
            allow_mask = torch.ones((num_nodes, num_nodes), device=device, dtype=torch.bool)
            allow_mask.fill_diagonal_(True)
            self._od_allow_mask = allow_mask
            self._od_allow_mask_num_nodes = num_nodes
            self._od_allow_mask_num_edges = num_edges
            return

        if (
            self._od_allow_mask is not None
            and self._od_allow_mask.device == device
            and self._od_allow_mask_num_nodes == num_nodes
            and self._od_allow_mask_num_edges == num_edges
        ):
            return

        src, dst = g_od.edges()
        src = src.to(device)
        dst = dst.to(device)

        if "flow" in g_od.edata:
            flow = g_od.edata["flow"].to(device).view(-1).float()
        else:
            flow = torch.ones((src.numel(),), device=device, dtype=torch.float32)

        neighbor_mask = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)

        if self.od_mask_topk == -1:
            neighbor_mask[src, dst] = True
            neighbor_mask[dst, src] = True
        else:
            u = torch.cat([src, dst], dim=0)
            v = torch.cat([dst, src], dim=0)
            w = torch.cat([flow, flow], dim=0)

            order = torch.argsort(u)
            u_sorted = u[order]
            v_sorted = v[order]
            w_sorted = w[order]

            start = 0
            for node in range(num_nodes):
                if start >= u_sorted.numel():
                    break
                if u_sorted[start].item() > node:
                    continue
                end = start
                while end < u_sorted.numel() and u_sorted[end].item() == node:
                    end += 1
                if end > start:
                    cand_v = v_sorted[start:end]
                    cand_w = w_sorted[start:end]
                    keep = cand_v != node
                    cand_v = cand_v[keep]
                    cand_w = cand_w[keep]
                    if cand_v.numel() > 0:
                        k = min(self.od_mask_topk, cand_v.numel())
                        top_idx = torch.topk(cand_w, k=k, largest=True).indices
                        top_v = cand_v[top_idx]
                        neighbor_mask[node, top_v] = True
                start = end

        neighbor_mask = neighbor_mask | neighbor_mask.t()
        neighbor_mask.fill_diagonal_(True)

        allow_mask = ~neighbor_mask
        allow_mask.fill_diagonal_(True)

        self._od_allow_mask = allow_mask
        self._od_allow_mask_num_nodes = num_nodes
        self._od_allow_mask_num_edges = num_edges

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        allow_mask = None
        if self._spatial_allow_mask is not None and self._od_allow_mask is not None:
            allow_mask = self._spatial_allow_mask & self._od_allow_mask
        elif self._spatial_allow_mask is not None:
            allow_mask = self._spatial_allow_mask
        elif self._od_allow_mask is not None:
            allow_mask = self._od_allow_mask
        loss = self._ntxent(z1, z2, self.tau, allow_mask=allow_mask)
        return loss, {"loss": loss.item()}

    def _ntxent(self, z1: torch.Tensor, z2: torch.Tensor, tau: float, allow_mask: torch.Tensor | None = None):
        logits = torch.matmul(z1, z2.t()) / tau
        if allow_mask is not None:
            logits = logits.masked_fill(~allow_mask, float("-inf"))
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

