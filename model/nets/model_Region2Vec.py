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


class UrbanModel(nn.Module):
    """
    两级结构：空间 GraphSAGE + OD 交叉注意力 + 投影头
    使用 Region2Vec-style community-oriented loss
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
        # Region2Vec-style loss 参数
        spatial_lambda: float = 0.1,  # 空间约束权重（已弃用，保留兼容性）
        loss_eps: float = 1e-15,  # 损失计算中的 epsilon（与 Region2Vec 一致）
        hops_threshold: int = 5,  # 跳距阈值，小于此值的跳距在约束中设为 0
        hops_matrix_path: str | None = None,  # 跳距矩阵文件路径（必须提供）
        loss_type: str = "divreg",  # 损失类型：'div' 或 'divreg'
    ):
        super().__init__()
        self.dims = dims
        in_dim = dims["poi"] + dims["res"] + \
            dims["vis"] + dims.get("street", 0)

        # Region2Vec loss 参数
        self.spatial_lambda = spatial_lambda  # 保留但不再使用
        self.loss_eps = loss_eps
        self.hops_threshold = hops_threshold
        self.hops_matrix_path = hops_matrix_path
        self.loss_type = loss_type

        # 验证跳距矩阵路径是否提供
        if not self.hops_matrix_path:
            raise ValueError("必须提供 hops_matrix_path 参数，跳距矩阵文件路径不能为空")

        # 缓存空间跳距权重矩阵（避免重复计算）
        self._hops_weight_matrix: torch.Tensor | None = None
        self._hops_weight_matrix_num_nodes: int | None = None

        self.spatial = SpatialSAGE(
            in_dim, hidden_dim, num_layers=sage_layers, dropout=dropout)

        self.q_dim = dims["res"]
        self.poi_dim = dims["poi"]
        self.vis_dim = dims["vis"]
        self.street_dim = dims.get("street", 0)
        self.fused_dim = self.poi_dim + self.vis_dim + self.street_dim

        self.attn = ODCrossAttention(
            in_q_dim=self.q_dim,
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
        self.q_proj = nn.Linear(hidden_dim, self.q_dim)
        self.poi_proj = nn.Linear(hidden_dim, self.poi_dim)
        self.vis_proj = nn.Linear(hidden_dim, self.vis_dim)
        self.street_proj = nn.Linear(
            hidden_dim, self.street_dim) if self.street_dim > 0 else None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # 直接进行一次编码（不使用视图增强）
        h_spatial, h_od, z = self._encode(batch)

        # 计算 Region2Vec-style loss
        loss, info = self._region2vec_loss(
            z, batch["g_od"], device=z.device
        )

        info.update({
            "h_spatial": h_spatial,
            "h_od": h_od,
            "z": z
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

        # 使用第一层输出构造 Q/K：Q 用 res 子空间，K 用 poi+vis+street 子空间（仅用于计算注意力分数）
        # Value 使用整体特征 h_spatial
        q_in = self.q_proj(h_spatial)
        poi_in = self.poi_proj(h_spatial)
        vis_in = self.vis_proj(h_spatial)
        if self.street_proj is not None:
            street_in = self.street_proj(h_spatial)
            fused_in = torch.cat([poi_in, vis_in, street_in], dim=-1)
        else:
            street_in = torch.zeros(
                (h_spatial.shape[0], 0), device=h_spatial.device, dtype=h_spatial.dtype)
            fused_in = torch.cat([poi_in, vis_in], dim=-1)

        h_od = self.attn(g_od, q_in, poi_in, vis_in, street_in,
                         fused_in, g_od.edata["flow"], h_spatial)
        h_od = self.ffn_od(h_od)

        h = h_spatial + h_od
        z = F.normalize(self.proj(h), dim=-1)
        return h_spatial, h_od, z

    def _load_hops_weight_matrix(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        从文件加载空间跳距权重矩阵。

        Returns:
            hops_weight_matrix: [N, N] float tensor，权重矩阵
        """
        # 若缓存有效且device一致，直接复用
        if (
            self._hops_weight_matrix is not None
            and self._hops_weight_matrix.device == device
            and self._hops_weight_matrix_num_nodes == num_nodes
        ):
            return self._hops_weight_matrix

        # 从文件加载跳距矩阵
        if not os.path.exists(self.hops_matrix_path):
            raise FileNotFoundError(f"跳距矩阵文件不存在: {self.hops_matrix_path}")

        import numpy as np
        hops_m = np.loadtxt(self.hops_matrix_path, delimiter=',')
        assert hops_m.shape[0] == hops_m.shape[1] == num_nodes, \
            f"跳距矩阵尺寸不匹配: 期望 {num_nodes}x{num_nodes}, 实际 {hops_m.shape}"

        # 转换为权重形式：hops_m = 1/(log(hops_m + EPS) + 1)
        zero_entries = hops_m < self.hops_threshold
        hops_m_weighted = 1 / (np.log(hops_m + self.loss_eps) + 1)
        hops_m_weighted[zero_entries] = 0

        # 调试信息：打印跳距统计（仅第一次加载时）
        if self._hops_weight_matrix is None:
            num_valid_hops = (~zero_entries).sum()
            if num_valid_hops > 0:
                print(f"[跳距矩阵] 文件={self.hops_matrix_path}, 阈值={self.hops_threshold}, "
                      f"有效节点对数={num_valid_hops}/{num_nodes*num_nodes}, "
                      f"跳距范围=[{hops_m[~zero_entries].min():.1f}, {hops_m[~zero_entries].max():.1f}], "
                      f"权重范围=[{hops_m_weighted[~zero_entries].min():.4f}, {hops_m_weighted[~zero_entries].max():.4f}]")
            else:
                print(f"[警告] 跳距矩阵中没有任何有效节点对（阈值={self.hops_threshold}），空间约束将无效")

        hops_weight_matrix = torch.FloatTensor(hops_m_weighted).to(device)

        # 缓存结果
        self._hops_weight_matrix = hops_weight_matrix
        self._hops_weight_matrix_num_nodes = num_nodes

        return hops_weight_matrix

    def _region2vec_loss(
        self,
        z: torch.Tensor,
        g_od: dgl.DGLGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Region2Vec-style community-oriented loss
        参考 Region2Vec (SIGSPATIAL 2022) 的实现

        Args:
            z: [N, D] node embeddings (已 L2 归一化)
            g_od: OD 图（包含 flow 边特征）
            device: 设备

        Returns:
            loss: 总损失
            info: 损失信息字典
        """
        num_nodes = z.size(0)

        # 1. 构建 flow 矩阵和 mask（从 OD 图）
        src, dst = g_od.edges()
        src = src.to(device)
        dst = dst.to(device)

        # 初始化 flow 矩阵（使用原始 flow 值）
        flow_matrix_raw = torch.zeros(
            (num_nodes, num_nodes), device=device, dtype=torch.float32)

        if src.numel() > 0:
            # 优先使用原始 flow 值（如果数据集提供了）
            if "flow_raw" in g_od.edata:
                flow_raw = g_od.edata["flow_raw"].to(device).view(-1).float()
            else:
                # 如果没有原始值，尝试从标准化值还原（近似）
                flow_normalized = g_od.edata["flow"].to(
                    device).view(-1).float()
                # 假设标准化是 (log1p(x) - mean) / std，我们需要近似还原
                # 这里使用 exp 来近似还原 log1p（注意：这只是近似，可能不准确）
                flow_raw = torch.exp(
                    flow_normalized.clamp(min=-10, max=10)) - 1.0
                flow_raw = flow_raw.clamp(min=0.0)  # 确保非负

            # 对称化：flow(i,j) = flow(j,i) = max(flow_ij, flow_ji)
            flow_matrix_raw[src, dst] = torch.maximum(
                flow_matrix_raw[src, dst],
                flow_raw
            )
            flow_matrix_raw[dst, src] = flow_matrix_raw[src, dst]

        # 构建正负样本 mask
        pos_mask = (flow_matrix_raw > 0).float()  # flow > 0
        neg_mask = (flow_matrix_raw == 0).float()  # flow == 0
        neg_mask.fill_diagonal_(0)  # 排除自身

        # Flow 权重：使用 log(flow + EPS)，与 Region2Vec 一致
        flow_labels = torch.log(flow_matrix_raw + self.loss_eps)

        # 统计正负样本数量
        N_pos = pos_mask.sum().item()
        N_neg = neg_mask.sum().item()

        # 2. 计算 embedding 距离矩阵（L2 距离）
        z_expanded_i = z.unsqueeze(1)  # [N, 1, D]
        z_expanded_j = z.unsqueeze(0)  # [1, N, D]
        pdist = torch.norm(z_expanded_i - z_expanded_j, dim=2, p=2)  # [N, N]

        # 3. 加载跳距权重矩阵（从文件）
        hops_weight_matrix = self._load_hops_weight_matrix(num_nodes, device)

        # 4. 计算空间约束项
        loss_hops = torch.sum(pdist * hops_weight_matrix) + self.loss_eps
        num_hops_pairs = (hops_weight_matrix > 0).sum().item()

        # 5. 计算损失（按照 Region2Vec 的 divreg 形式）
        if self.loss_type == "div":
            # div 形式：loss = sum(pdist * labels * pos_mask) / (sum(pdist * neg_mask) + loss_hops)
            loss_train = torch.sum(pdist * flow_labels * pos_mask) / (
                torch.sum(pdist * neg_mask) + loss_hops
            )
        elif self.loss_type == "divreg":
            # divreg 形式（带归一化）：loss = sum(pdist * labels * pos_mask) * N_neg / (N_pos * (sum(pdist * neg_mask) + loss_hops))
            if N_pos > 0 and N_neg > 0:
                loss_train = torch.sum(pdist * flow_labels * pos_mask) * N_neg / (
                    N_pos * (torch.sum(pdist * neg_mask) + loss_hops)
                )
            else:
                # 如果没有正样本或负样本，使用简化形式
                loss_train = torch.sum(pdist * flow_labels * pos_mask) / (
                    torch.sum(pdist * neg_mask) + loss_hops + self.loss_eps
                )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 计算各项损失的详细值（用于调试和监控）
        loss_pos_value = torch.sum(
            pdist * flow_labels * pos_mask).item() if N_pos > 0 else 0.0
        loss_neg_value = torch.sum(
            pdist * neg_mask).item() if N_neg > 0 else 0.0

        info = {
            "loss": loss_train.item(),
            # 正样本项：sum(pdist * log(flow) * pos_mask)
            "loss_pos": loss_pos_value,
            "loss_neg": loss_neg_value,  # 负样本项：sum(pdist * neg_mask)
            "loss_hops": loss_hops.item(),  # 空间约束项：sum(pdist * hops_weight_matrix)
            "loss_spatial": loss_hops.item(),  # 别名，用于兼容训练脚本
            "num_pos_pairs": int(N_pos),  # 正样本对数量（flow > 0）
            "num_neg_pairs": int(N_neg),  # 负样本对数量（flow == 0）
            # 参与空间约束的节点对数量（跳距 >= threshold）
            "num_hops_pairs": int(num_hops_pairs),
            "hops_weight_sum": hops_weight_matrix.sum().item(),  # 跳距权重矩阵的总和
        }

        return loss_train, info
