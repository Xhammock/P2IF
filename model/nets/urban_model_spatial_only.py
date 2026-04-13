from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

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


class UrbanModelSpatialOnly(nn.Module):
    """
    仅使用空间邻接矩阵的GraphSAGE网络（消融实验）
    - 只使用空间GraphSAGE，不包含OD层
    - 保留视图增强的对比学习（Spatial-aware NT-Xent）
    - 支持灵活的特征选择（poi, res, vis, street都可以开关）
    - 只输出z（投影后的表征）
    """

    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 256,
        sage_layers: int = 1,
        dropout: float = 0.1,
        proj_dim: int = 128,
        # 视图增强参数
        feat_drop_ratio: float = 0.1,
        edge_drop_ratio: float = 0.075,
        noise_std: float = 0.01,
        tau: float = 0.1,  # NT-Xent temperature
        # 消融实验参数：控制是否使用某个特征模态
        use_poi: bool = True,
        use_res: bool = True,
        use_vis: bool = True,
        use_street: bool = True,
    ):
        super().__init__()
        self.dims = dims
        
        # 消融实验：根据use_poi、use_res、use_vis和use_street参数调整实际使用的维度
        self.use_poi = use_poi
        self.use_res = use_res
        self.use_vis = use_vis
        self.use_street = use_street
        
        actual_poi_dim = dims["poi"] if use_poi else 0
        actual_res_dim = dims["res"] if use_res else 0
        actual_vis_dim = dims["vis"] if use_vis else 0
        actual_street_dim = dims.get("street", 0) if use_street else 0
        
        in_dim = actual_poi_dim + actual_res_dim + actual_vis_dim + actual_street_dim
        
        if in_dim == 0:
            raise ValueError("至少需要启用一个特征模态（poi/res/vis/street）")
        
        # 保存原始维度信息，用于从完整特征中提取需要的部分
        self.original_poi_dim = dims["poi"]
        self.original_res_dim = dims["res"]
        self.original_vis_dim = dims["vis"]
        self.original_street_dim = dims.get("street", 0)

        # 视图增强参数
        self.tau = tau
        self.feat_drop_ratio = feat_drop_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.noise_std = noise_std

        # Spatial-aware NT-Xent：缓存"允许作为负样本"的mask（非邻居=True，邻居/自身=False）
        # 注意：会在forward里根据传入的原始空间图懒构建，并随device迁移
        self._spatial_allow_mask: torch.Tensor | None = None
        self._spatial_allow_mask_num_nodes: int | None = None
        self._spatial_allow_mask_num_edges: int | None = None

        # 空间编码器
        self.spatial = SpatialSAGE(
            in_dim, hidden_dim, num_layers=sage_layers, dropout=dropout)
        
        # FeedForward层
        self.ffn = FeedForward(hidden_dim, dropout)
        
        # 投影头
        self.proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        训练时：生成两个增强视图，计算对比损失
        """
        # 生成两个增强视图
        view1_spatial, view1_feat = self._create_view1(
            batch["g_spatial"], batch["feat"])
        view2_spatial, view2_feat = self._create_view2(
            batch["g_spatial"], batch["feat"])

        # 对两个视图分别编码
        batch_v1 = {"g_spatial": view1_spatial, "feat": view1_feat}
        batch_v2 = {"g_spatial": view2_spatial, "feat": view2_feat}
        z_v1 = self._encode(batch_v1)
        z_v2 = self._encode(batch_v2)

        # Spatial-aware NT-Xent：使用"原始空间图"构建mask（不基于dropedge后的图）
        # - 空间邻居不作为负样本
        self._ensure_spatial_allow_mask(
            batch["g_spatial"], device=z_v1.device)

        # 计算对比损失（z_v1和z_v2已在_encode中L2归一化）
        loss, info = self._contrastive_loss(z_v1, z_v2)

        # 使用v1的编码作为主要输出（用于推理）
        info.update({
            "z": z_v1
        })
        return loss, info

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        返回最终投影后的表征 z（已 L2 归一化）。
        用于推理/可视化，外部需保证 no_grad。
        推理时不使用数据增强。
        """
        z = self._encode(batch)
        return z

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
        
        # Res特征
        if self.use_res:
            parts.append(feats[:, start_idx:start_idx + self.original_res_dim])
        start_idx += self.original_res_dim
        
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

    def _encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        编码函数：空间GraphSAGE -> FFN -> 投影 -> L2归一化
        返回：z [N, proj_dim]，已L2归一化
        """
        g_spatial = batch["g_spatial"]
        feats = batch["feat"]
        device = feats.device

        # 消融实验：从完整特征中提取需要的部分
        feats = self._extract_features(feats)

        # Spatial encoder
        h_spatial = self.spatial(g_spatial, feats)
        h_spatial = self.ffn(h_spatial)

        # 投影并L2归一化
        z = F.normalize(self.proj(h_spatial), dim=-1)
        return z

    def _create_view1(self, g_spatial, feats: torch.Tensor) -> Tuple:
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
        return view1_spatial, view1_feat

    def _create_view2(self, g_spatial, feats: torch.Tensor) -> Tuple:
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
        return view2_spatial, view2_feat

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

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        基于视图增强的对比损失
        - z1, z2: 两个视图的节点表征 [N, D]，已L2归一化
        - 对应节点互为正样本，使用in-batch negatives
        """
        allow_mask = self._spatial_allow_mask

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

