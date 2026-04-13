from typing import Dict, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np

from model.layers.spatial_sage import SpatialSAGE
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


class ODGAT(nn.Module):
    """
    基于 OD 图的传统 GAT 层
    使用 DGL 的 GATConv
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(
                dglnn.GATConv(
                    in_feats=dims[i],
                    out_feats=self.head_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=F.elu if i < num_layers - 1 else None,
                    allow_zero_in_degree=True,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            g: OD 图
            feats: [N, in_dim] 节点特征
            edge_weight: [E, 1] 边权重（flow），可选（当前版本未使用，保留接口兼容性）
        Returns:
            h: [N, hidden_dim] 输出特征
        """
        h = feats

        for i, layer in enumerate(self.layers):
            # GATConv 返回 [N, num_heads, head_dim]
            h = layer(g, h)
            # 如果不是最后一层，需要reshape以便下一层使用
            if i < len(self.layers) - 1:
                # 中间层：reshape 为 [N, hidden_dim]
                h = h.reshape(h.shape[0], -1)
                h = self.dropout(h)
            else:
                # 最后一层：reshape 为 [N, hidden_dim]
                h = h.reshape(h.shape[0], -1)

        return h


class UrbanModelGAT(nn.Module):
    """
    两级结构：空间 GraphSAGE + OD GAT + 投影头
    使用视图增强的对比学习（Spatial-aware NT-Xent）
    OD层使用传统GAT而非Transformer注意力
    """

    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 256,
        sage_layers: int = 1,
        gat_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        proj_dim: int = 128,
        loss_weight: Dict[str, float] | None = None,
        # 视图增强参数
        feat_drop_ratio: float = 0.1,
        edge_drop_ratio: float = 0.075,
        noise_std: float = 0.01,
        tau: float = 0.1,  # NT-Xent temperature
        # Spatial-aware NT-Xent：额外屏蔽 OD 强邻居
        # od_mask_topk == -1: 屏蔽所有有流量的节点对（flow > 0）
        # od_mask_topk > 0: 屏蔽每个节点流量最大的 top-k 个邻居
        # od_mask_topk <= 0 (且 != -1): 不屏蔽OD邻居
        od_mask_topk: int = 200,
    ):
        super().__init__()
        self.dims = dims
        in_dim = dims["poi"] + dims["res"] + \
            dims["vis"] + dims.get("street", 0)

        # 视图增强参数
        self.tau = tau
        self.feat_drop_ratio = feat_drop_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.noise_std = noise_std
        self.od_mask_topk = int(od_mask_topk)

        # Spatial-aware NT-Xent：缓存"允许作为负样本"的mask
        self._spatial_allow_mask: torch.Tensor | None = None
        self._spatial_allow_mask_num_nodes: int | None = None
        self._spatial_allow_mask_num_edges: int | None = None

        # OD allow mask
        self._od_allow_mask: torch.Tensor | None = None
        self._od_allow_mask_num_nodes: int | None = None
        self._od_allow_mask_num_edges: int | None = None

        # 空间编码器
        self.spatial = SpatialSAGE(
            in_dim, hidden_dim, num_layers=sage_layers, dropout=dropout)

        # OD层：使用传统GAT
        self.od_gat = ODGAT(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=gat_layers,
            num_heads=n_heads,
            dropout=dropout,
        )

        self.ffn_spatial = FeedForward(hidden_dim, dropout)
        self.ffn_od = FeedForward(hidden_dim, dropout)

        self.proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # 生成两个增强视图
        view1_spatial, view1_od, view1_feat = self._create_view1(
            batch["g_spatial"], batch["g_od"], batch["feat"])
        view2_spatial, view2_od, view2_feat = self._create_view2(
            batch["g_spatial"], batch["g_od"], batch["feat"])

        # 对两个视图分别编码
        batch_v1 = {"g_spatial": view1_spatial,
                    "g_od": view1_od, "feat": view1_feat}
        batch_v2 = {"g_spatial": view2_spatial,
                    "g_od": view2_od, "feat": view2_feat}
        h_spatial_v1, h_od_v1, z_v1 = self._encode(batch_v1)
        h_spatial_v2, h_od_v2, z_v2 = self._encode(batch_v2)

        # Spatial-aware NT-Xent：使用"原始空间图/OD图"构建mask
        self._ensure_spatial_allow_mask(
            batch["g_spatial"], device=z_v1.device)
        self._ensure_od_allow_mask(batch["g_od"], device=z_v1.device)

        # 计算对比损失
        loss, info = self._contrastive_loss(z_v1, z_v2)

        # 使用v1的编码作为主要输出
        info.update({
            "h_spatial": h_spatial_v1,
            "h_od": h_od_v1,
            "z": z_v1
        })
        return loss, info

    def encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回空间层输出、OD 层输出以及最终投影后的表征 z（已 L2 归一化）。
        用于推理/可视化，外部需保证 no_grad。
        推理时不使用数据增强。
        """
        h_spatial, h_od, z = self._encode(batch)
        return h_spatial, h_od, z

    def _encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_spatial = batch["g_spatial"]
        g_od = batch["g_od"]
        feats = batch["feat"]
        device = feats.device

        # Spatial encoder
        h_spatial = self.spatial(g_spatial, feats)
        h_spatial = self.ffn_spatial(h_spatial)

        # OD层：使用传统GAT
        # 获取边权重（flow）
        edge_flow = g_od.edata.get("flow", None)
        if edge_flow is not None:
            edge_flow = edge_flow.to(device)

        h_od = self.od_gat(g_od, h_spatial, edge_weight=edge_flow)
        h_od = self.ffn_od(h_od)

        h = h_spatial + h_od
        z = F.normalize(self.proj(h), dim=-1)
        return h_spatial, h_od, z

    def _create_view1(self, g_spatial, g_od, feats: torch.Tensor) -> Tuple:
        """
        视图1：特征dropout + 高斯噪声
        """
        if self.training and self.feat_drop_ratio > 0:
            feat_mask = torch.rand(
                feats.shape, device=feats.device) > self.feat_drop_ratio
            view1_feat = feats * feat_mask.float()
        else:
            view1_feat = feats

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(view1_feat) * self.noise_std
            view1_feat = view1_feat + noise

        view1_spatial = g_spatial.clone()
        view1_spatial.ndata["feat"] = view1_feat.clone()
        view1_od = g_od.clone()
        view1_od.ndata["feat"] = view1_feat.clone()
        return view1_spatial, view1_od, view1_feat

    def _create_view2(self, g_spatial, g_od, feats: torch.Tensor) -> Tuple:
        """
        视图2：dropedge + 高斯噪声
        """
        view2_feat = feats
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(view2_feat) * self.noise_std
            view2_feat = view2_feat + noise

        view2_spatial = self._drop_edges(g_spatial, self.edge_drop_ratio)
        view2_spatial.ndata["feat"] = view2_feat.clone()
        view2_od = self._drop_edges(g_od, self.edge_drop_ratio)
        view2_od.ndata["feat"] = view2_feat.clone()
        return view2_spatial, view2_od, view2_feat

    def _drop_edges(self, g, drop_ratio: float):
        """
        随机删除图中一定比例的边
        """
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
        src_keep = src[keep_eids]
        dst_keep = dst[keep_eids]
        new_g = dgl.graph((src_keep, dst_keep), num_nodes=g.num_nodes())
        new_g.ndata["feat"] = g.ndata["feat"].clone()

        if len(g.edata) > 0:
            for key in g.edata:
                new_g.edata[key] = g.edata[key][keep_eids].clone()

        return new_g

    def _ensure_spatial_allow_mask(self, g_spatial, device: torch.device) -> None:
        """
        构建/更新空间非邻居mask
        """
        num_nodes = g_spatial.num_nodes()
        num_edges = g_spatial.num_edges()

        if (
            self._spatial_allow_mask is not None
            and self._spatial_allow_mask.device == device
            and self._spatial_allow_mask_num_nodes == num_nodes
            and self._spatial_allow_mask_num_edges == num_edges
        ):
            return

        adj = torch.zeros((num_nodes, num_nodes),
                          device=device, dtype=torch.bool)
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
        """
        构建/更新 OD 非邻居mask
        定义 OD 邻居：
        - 如果 od_mask_topk == -1：屏蔽所有有流量的节点对（flow > 0）
        - 如果 od_mask_topk > 0：对每个节点 i，从与 i 相连的 OD 边（忽略方向）中，
          选择 flow 最大的 top-k 个邻居节点作为"OD 邻居"（不作为负样本）。
        - 如果 od_mask_topk <= 0（且 != -1）：不屏蔽OD邻居
        """
        num_nodes = g_od.num_nodes()
        num_edges = g_od.num_edges()

        if self.od_mask_topk <= 0 and self.od_mask_topk != -1:
            allow_mask = torch.ones(
                (num_nodes, num_nodes), device=device, dtype=torch.bool)
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
            flow = torch.ones((src.numel(),), device=device,
                              dtype=torch.float32)

        neighbor_mask = torch.zeros(
            (num_nodes, num_nodes), device=device, dtype=torch.bool)

        # od_mask_topk == -1：屏蔽所有有流量的节点对
        if self.od_mask_topk == -1:
            # 直接标记所有有边的节点对
            neighbor_mask[src, dst] = True
            neighbor_mask[dst, src] = True  # 无向图
        else:
            # 逐节点分段取 top-k
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

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        基于视图增强的对比损失
        """
        allow_mask = None
        if self._spatial_allow_mask is not None and self._od_allow_mask is not None:
            allow_mask = self._spatial_allow_mask & self._od_allow_mask
        elif self._spatial_allow_mask is not None:
            allow_mask = self._spatial_allow_mask
        elif self._od_allow_mask is not None:
            allow_mask = self._od_allow_mask

        loss = self._ntxent(z1, z2, self.tau, allow_mask=allow_mask)
        return loss, {"loss": loss.item()}

    def _ntxent(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        tau: float,
        allow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Standard vectorized NT-Xent loss (in-batch negatives)
        """
        logits = torch.matmul(z1, z2.t()) / tau  # [N, N]

        if allow_mask is not None:
            logits = logits.masked_fill(~allow_mask, float("-inf"))

        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)
