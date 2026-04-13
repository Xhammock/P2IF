from typing import Dict, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np

from model.layers.spatial_sage import SpatialSAGE
from model.layers.od_cross_attention import ODCrossAttention
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


class UrbanModelAugFullQuery(nn.Module):
    """
    两级结构：空间 GraphSAGE + OD 交叉注意力 + 投影头
    使用视图增强的对比学习（Spatial-aware NT-Xent）
    
    消融实验变体：使用完整的 embedding (h_spatial) 作为查询向量，
    而不是使用 res 子空间，这样可以去掉 res 部分的影响。
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
        # 消融实验参数：控制是否使用某个特征模态
        use_poi: bool = True,
        use_vis: bool = True,
        use_street: bool = True,
    ):
        super().__init__()
        self.dims = dims
        # 消融实验：根据use_poi、use_vis和use_street参数调整实际使用的维度
        self.use_poi = use_poi
        self.use_vis = use_vis
        self.use_street = use_street
        actual_poi_dim = dims["poi"] if use_poi else 0
        actual_vis_dim = dims["vis"] if use_vis else 0
        actual_street_dim = dims.get("street", 0) if use_street else 0
        in_dim = actual_poi_dim + dims["res"] + actual_vis_dim + actual_street_dim
        
        # 保存原始维度信息，用于从完整特征中提取需要的部分
        self.original_poi_dim = dims["poi"]
        self.original_vis_dim = dims["vis"]
        self.original_street_dim = dims.get("street", 0)
        self.res_dim = dims["res"]
        self.street_dim = actual_street_dim

        # 视图增强参数
        self.tau = tau
        self.feat_drop_ratio = feat_drop_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.noise_std = noise_std
        self.od_mask_topk = int(od_mask_topk)

        # Spatial-aware NT-Xent：缓存"允许作为负样本"的mask（非邻居=True，邻居/自身=False）
        # 注意：会在forward里根据传入的原始空间图懒构建，并随device迁移
        self._spatial_allow_mask: torch.Tensor | None = None
        self._spatial_allow_mask_num_nodes: int | None = None
        self._spatial_allow_mask_num_edges: int | None = None

        # OD allow mask（按每个节点 top-k flow 选邻居并屏蔽）
        self._od_allow_mask: torch.Tensor | None = None
        self._od_allow_mask_num_nodes: int | None = None
        self._od_allow_mask_num_edges: int | None = None

        self.spatial = SpatialSAGE(
            in_dim, hidden_dim, num_layers=sage_layers, dropout=dropout)

        # 关键修改：使用完整的 hidden_dim 作为查询维度，而不是 res 子空间
        self.q_dim = hidden_dim
        # 使用实际维度（考虑消融实验）
        self.poi_dim = actual_poi_dim
        self.vis_dim = actual_vis_dim
        self.street_dim = actual_street_dim
        self.fused_dim = self.poi_dim + self.vis_dim + self.street_dim

        self.attn = ODCrossAttention(
            in_q_dim=self.q_dim,  # 使用 hidden_dim 而不是 res_dim
            in_poi_dim=self.poi_dim,
            in_vis_dim=self.vis_dim,
            in_street_dim=self.street_dim,
            in_fused_dim=self.fused_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.ffn_spatial = FeedForward(hidden_dim, dropout)
        self.ffn_od = FeedForward(hidden_dim, dropout)

        self.proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)
        # 将空间层输出映射到 Query / 各模态 KeyValue 所需维度
        # 关键修改：q_proj 从 hidden_dim 映射到 hidden_dim（完整embedding）
        self.q_proj = nn.Linear(hidden_dim, self.q_dim)  # hidden_dim -> hidden_dim
        self.poi_proj = nn.Linear(hidden_dim, self.poi_dim) if self.poi_dim > 0 else None
        self.vis_proj = nn.Linear(hidden_dim, self.vis_dim) if self.vis_dim > 0 else None
        self.street_proj = nn.Linear(
            hidden_dim, self.street_dim) if self.street_dim > 0 else None

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

        # Spatial-aware NT-Xent：使用"原始空间图/OD图"构建mask（不基于dropedge后的图）
        # - 空间邻居不作为负样本
        # - OD 强邻居（每个节点 top-k flow）不作为负样本
        self._ensure_spatial_allow_mask(
            batch["g_spatial"], device=z_v1.device)
        self._ensure_od_allow_mask(batch["g_od"], device=z_v1.device)

        # 计算对比损失（z_v1和z_v2已在_encode中L2归一化）
        loss, info = self._contrastive_loss(z_v1, z_v2)

        # 使用v1的编码作为主要输出（用于推理）
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

    def _extract_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        从完整的特征向量中提取需要的部分（支持消融实验）。
        完整特征格式：[poi, res, vis, street]
        """
        parts = []
        start_idx = 0
        
        # POI特征
        if self.use_poi:
            parts.append(feats[:, start_idx:start_idx + self.original_poi_dim])
        start_idx += self.original_poi_dim
        
        # Res特征（总是使用，因为需要输入到spatial encoder）
        parts.append(feats[:, start_idx:start_idx + self.res_dim])
        start_idx += self.res_dim
        
        # Vis特征
        if self.use_vis:
            parts.append(feats[:, start_idx:start_idx + self.original_vis_dim])
        # 无论是否使用vis，都需要跳过这部分索引（如果原始数据中有vis特征）
        start_idx += self.original_vis_dim
        
        # Street特征（街景）
        if self.use_street and self.original_street_dim > 0:
            parts.append(feats[:, start_idx:start_idx + self.original_street_dim])
        # 无论是否使用street，都需要跳过这部分索引（如果原始数据中有street特征）
        if self.original_street_dim > 0:
            start_idx += self.original_street_dim
        
        return torch.cat(parts, dim=-1) if len(parts) > 0 else feats

    def _encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_spatial = batch["g_spatial"]
        g_od = batch["g_od"]
        feats = batch["feat"]
        device = feats.device

        # 消融实验：从完整特征中提取需要的部分
        feats = self._extract_features(feats)

        # Spatial encoder
        h_spatial = self.spatial(g_spatial, feats)
        h_spatial = self.ffn_spatial(h_spatial)

        # 关键修改：使用完整的 h_spatial 作为查询向量，而不是 res 子空间
        # Q 使用完整的 embedding (h_spatial)，K 用 poi+vis+street 子空间（仅用于计算注意力分数）
        # Value 使用整体特征 h_spatial
        q_in = self.q_proj(h_spatial)  # hidden_dim -> hidden_dim（完整embedding）
        
        if self.poi_proj is not None:
            poi_in = self.poi_proj(h_spatial)
        else:
            poi_in = torch.zeros(
                (h_spatial.shape[0], 0), device=h_spatial.device, dtype=h_spatial.dtype)
        
        if self.vis_proj is not None:
            vis_in = self.vis_proj(h_spatial)
        else:
            vis_in = torch.zeros(
                (h_spatial.shape[0], 0), device=h_spatial.device, dtype=h_spatial.dtype)
        
        if self.street_proj is not None:
            street_in = self.street_proj(h_spatial)
        else:
            street_in = torch.zeros(
                (h_spatial.shape[0], 0), device=h_spatial.device, dtype=h_spatial.dtype)
        
        # 构建fused特征：拼接所有可用的模态
        fused_parts = []
        if self.poi_dim > 0:
            fused_parts.append(poi_in)
        if self.vis_dim > 0:
            fused_parts.append(vis_in)
        if self.street_dim > 0:
            fused_parts.append(street_in)
        
        if len(fused_parts) > 0:
            fused_in = torch.cat(fused_parts, dim=-1)
        else:
            # 如果所有模态都被禁用，创建一个空的fused特征
            fused_in = torch.zeros(
                (h_spatial.shape[0], 0), device=h_spatial.device, dtype=h_spatial.dtype)

        h_od = self.attn(g_od, q_in, poi_in, vis_in, street_in,
                         fused_in, g_od.edata["flow"], h_spatial)
        h_od = self.ffn_od(h_od)

        h = h_spatial + h_od
        z = F.normalize(self.proj(h), dim=-1)
        return h_spatial, h_od, z

    def _create_view1(self, g_spatial, g_od, feats: torch.Tensor) -> Tuple:
        """
        视图1：特征dropout + 高斯噪声
        """
        # 特征dropout：随机mask一些特征维度
        if self.training and self.feat_drop_ratio > 0:
            feat_mask = torch.rand(
                feats.shape, device=feats.device) > self.feat_drop_ratio
            view1_feat = feats * feat_mask.float()
        else:
            view1_feat = feats

        # 添加高斯噪声
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(view1_feat) * self.noise_std
            view1_feat = view1_feat + noise

        # 图结构不变，只更新特征
        view1_spatial = g_spatial.clone()
        view1_spatial.ndata["feat"] = view1_feat.clone()
        view1_od = g_od.clone()
        view1_od.ndata["feat"] = view1_feat.clone()
        return view1_spatial, view1_od, view1_feat

    def _create_view2(self, g_spatial, g_od, feats: torch.Tensor) -> Tuple:
        """
        视图2：dropedge (5%-10%) + 高斯噪声
        """
        # 添加高斯噪声到特征
        view2_feat = feats
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(view2_feat) * self.noise_std
            view2_feat = view2_feat + noise

        # DropEdge：随机删除一些边
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

        # 获取设备信息（从节点特征获取）
        device = g.ndata["feat"].device

        # 随机选择要保留的边
        eids = torch.randperm(num_edges, device=device)
        keep_eids = eids[num_drop:].sort()[0]

        # 创建新图，只保留选中的边
        src, dst = g.edges()
        src_keep = src[keep_eids]
        dst_keep = dst[keep_eids]
        new_g = dgl.graph((src_keep, dst_keep), num_nodes=g.num_nodes())
        new_g.ndata["feat"] = g.ndata["feat"].clone()

        # 保留边特征（如果有）
        if len(g.edata) > 0:
            for key in g.edata:
                new_g.edata[key] = g.edata[key][keep_eids].clone()

        return new_g

    def _ensure_spatial_allow_mask(self, g_spatial, device: torch.device) -> None:
        """
        构建/更新空间非邻居mask（allow_mask），用于在NT-Xent里屏蔽空间邻居负样本。
        allow_mask[i, j] == True 表示：j 可以作为 i 的负样本（即 i 与 j 在空间图中不相邻）。
        注意：为了不影响正样本（labels=arange(N)），对角线会强制为 True。
        """
        num_nodes = g_spatial.num_nodes()
        num_edges = g_spatial.num_edges()

        # 若缓存有效且device一致，直接复用
        if (
            self._spatial_allow_mask is not None
            and self._spatial_allow_mask.device == device
            and self._spatial_allow_mask_num_nodes == num_nodes
            and self._spatial_allow_mask_num_edges == num_edges
        ):
            return

        # 构建邻接（含自环），再取非邻居
        adj = torch.zeros((num_nodes, num_nodes),
                          device=device, dtype=torch.bool)
        src, dst = g_spatial.edges()
        src = src.to(device)
        dst = dst.to(device)
        adj[src, dst] = True
        adj[dst, src] = True  # 无向图
        adj.fill_diagonal_(True)  # 自身不作为负样本

        allow_mask = ~adj  # True表示非邻居
        allow_mask.fill_diagonal_(True)  # 但对角线必须保留为True，保证正样本可用

        self._spatial_allow_mask = allow_mask
        self._spatial_allow_mask_num_nodes = num_nodes
        self._spatial_allow_mask_num_edges = num_edges

    def _ensure_od_allow_mask(self, g_od, device: torch.device) -> None:
        """
        构建/更新 OD 非邻居mask（allow_mask），用于在NT-Xent里屏蔽 OD 强邻居负样本。
        定义 OD 邻居：对每个节点 i，从与 i 相连的 OD 边（忽略方向）中，
        选择 flow 最大的 top-k 个邻居节点作为"OD 邻居"（不作为负样本）。
        allow_mask_od[i, j] == True 表示：j 可以作为 i 的负样本（即 j 不是 i 的 top-k OD 邻居）。
        对角线强制为 True（保证正样本可用）。
        """
        num_nodes = g_od.num_nodes()
        num_edges = g_od.num_edges()

        # od_mask_topk == -1: 屏蔽所有有流量的节点对
        if self.od_mask_topk == -1:
            # 构建所有有流量的节点对mask
            src, dst = g_od.edges()
            src = src.to(device)
            dst = dst.to(device)
            
            neighbor_mask = torch.zeros(
                (num_nodes, num_nodes), device=device, dtype=torch.bool)
            neighbor_mask[src, dst] = True
            neighbor_mask[dst, src] = True  # 对称化
            neighbor_mask.fill_diagonal_(True)  # 自身不作为负样本
            
            allow_mask = ~neighbor_mask
            allow_mask.fill_diagonal_(True)  # 但对角线必须保留为True，保证正样本可用
            
            self._od_allow_mask = allow_mask
            self._od_allow_mask_num_nodes = num_nodes
            self._od_allow_mask_num_edges = num_edges
            return

        # k<=0：不屏蔽OD邻居
        if self.od_mask_topk <= 0:
            allow_mask = torch.ones(
                (num_nodes, num_nodes), device=device, dtype=torch.bool)
            allow_mask.fill_diagonal_(True)
            self._od_allow_mask = allow_mask
            self._od_allow_mask_num_nodes = num_nodes
            self._od_allow_mask_num_edges = num_edges
            return

        # 缓存有效且device一致，直接复用
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

        # 忽略方向：同时加入 src->dst 与 dst->src
        u = torch.cat([src, dst], dim=0)
        v = torch.cat([dst, src], dim=0)
        w = torch.cat([flow, flow], dim=0)

        order = torch.argsort(u)
        u_sorted = u[order]
        v_sorted = v[order]
        w_sorted = w[order]

        neighbor_mask = torch.zeros(
            (num_nodes, num_nodes), device=device, dtype=torch.bool)

        # 逐节点分段取 top-k（城市网格规模通常可接受；需要更大规模可再做向量化/稀疏化）
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

                # 去掉自身
                keep = cand_v != node
                cand_v = cand_v[keep]
                cand_w = cand_w[keep]

                if cand_v.numel() > 0:
                    k = min(self.od_mask_topk, cand_v.numel())
                    top_idx = torch.topk(cand_w, k=k, largest=True).indices
                    top_v = cand_v[top_idx]
                    neighbor_mask[node, top_v] = True
            start = end

        # 对称化（OD 邻居视作无向）
        neighbor_mask = neighbor_mask | neighbor_mask.t()
        neighbor_mask.fill_diagonal_(True)  # 自身不作为负样本

        allow_mask = ~neighbor_mask
        allow_mask.fill_diagonal_(True)  # 但对角线必须保留为True，保证正样本可用

        self._od_allow_mask = allow_mask
        self._od_allow_mask_num_nodes = num_nodes
        self._od_allow_mask_num_edges = num_edges

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        基于视图增强的对比损失
        - z1, z2: 两个视图的节点表征 [N, D]，已L2归一化
        - 对应节点互为正样本，使用in-batch negatives
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

        Args:
            z1, z2: [N, D], L2-normalized node representations
            tau: temperature
            allow_mask: [N, N] bool，True表示该位置参与softmax；False表示屏蔽（置为-inf）

        Returns:
            contrastive loss
        """
        logits = torch.matmul(z1, z2.t()) / tau  # [N, N]

        if allow_mask is not None:
            # 屏蔽"自身+空间邻居"作为负样本（但对角线必须是True，否则labels无法对齐）
            logits = logits.masked_fill(~allow_mask, float("-inf"))

        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

