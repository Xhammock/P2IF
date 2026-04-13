from typing import Dict, Tuple

import torch
import torch.nn as nn

from model.nets.model_aug import UrbanModelAug


class UrbanModelAugWithoutVis(UrbanModelAug):
    """
    消融实验变体：W/o Vis（去除访客画像信息）。

    实现方式：复用主模型结构，但强制在交互层禁用 vis 模态（use_vis=False）。
    其余模块保持一致。
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
        use_street: bool = True,
    ):
        super().__init__(
            dims=dims,
            hidden_dim=hidden_dim,
            sage_layers=sage_layers,
            n_heads=n_heads,
            dropout=dropout,
            proj_dim=proj_dim,
            loss_weight=loss_weight,
            feat_drop_ratio=feat_drop_ratio,
            edge_drop_ratio=edge_drop_ratio,
            noise_std=noise_std,
            tau=tau,
            od_mask_topk=od_mask_topk,
            use_poi=use_poi,
            use_vis=False,
            use_street=use_street,
        )

