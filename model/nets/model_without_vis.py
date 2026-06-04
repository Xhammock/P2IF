from typing import Dict, Tuple

import torch
import torch.nn as nn

from model.nets.model_aug import UrbanModelAug


class UrbanModelAugWithoutVis(UrbanModelAug):
    """
    Ablation variant: W/o Vis (removes visitor profile information).

    Implementation: reuse the main model architecture but force use_vis=False in the
    interaction layer. All other modules remain unchanged.
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
