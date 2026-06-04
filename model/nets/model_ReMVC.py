from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.spatial_sage import SpatialSAGE
from model.layers.projection_head import ProjectionHead


class UrbanModelReMVC(nn.Module):
    """
    ReMVC baseline (implementation aligned with this repository's data interface):
    - POI view: encode POI features on the spatial adjacency graph g_spatial -> z_poi
    - Flow view: encode full features (profiles + POI + street) on OD graph g_od -> z_flow
    - Training objective: cross-view contrastive learning (InfoNCE / NT-Xent); node i is a
      positive pair across the two views
    - Output embedding: concat(z_flow, z_poi) as input to downstream predictors
    """

    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 256,
        sage_layers: int = 2,
        dropout: float = 0.1,
        proj_dim: int = 128,
        tau: float = 0.1,
        use_street: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self.tau = tau

        self.poi_dim = int(dims["poi"])
        self.res_dim = int(dims["res"])
        self.vis_dim = int(dims["vis"])
        self.street_dim = int(dims.get("street", 0)) if use_street else 0

        self.full_in_dim = self.poi_dim + self.res_dim + self.vis_dim + self.street_dim

        # POI view: POI sub-vector only
        self.poi_encoder = SpatialSAGE(
            in_dim=self.poi_dim,
            hidden_dim=hidden_dim,
            num_layers=sage_layers,
            dropout=dropout,
        )
        self.poi_proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)

        # Flow view: full features aggregated on the OD graph
        self.flow_encoder = SpatialSAGE(
            in_dim=self.full_in_dim,
            hidden_dim=hidden_dim,
            num_layers=sage_layers,
            dropout=dropout,
        )
        self.flow_proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        z_flow, z_poi, z = self.encode(batch)
        loss = self._cross_view_ntxent(z_poi, z_flow, tau=self.tau)
        return loss, {"loss": loss.item(), "z": z, "z_poi": z_poi, "z_flow": z_flow}

    def encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_spatial = batch["g_spatial"]
        g_od = batch["g_od"]
        feats = batch["feat"]

        poi_feats = feats[:, : self.poi_dim]

        # When use_street=False, drop trailing street dims so feats align with full_in_dim
        if self.street_dim == 0 and "street" in self.dims and int(self.dims.get("street", 0)) > 0:
            feats_full = feats[:, : self.poi_dim + self.res_dim + self.vis_dim]
        else:
            feats_full = feats[:, : self.full_in_dim]

        h_poi = self.poi_encoder(g_spatial, poi_feats)
        h_flow = self.flow_encoder(g_od, feats_full)

        z_poi = F.normalize(self.poi_proj(h_poi), dim=-1)
        z_flow = F.normalize(self.flow_proj(h_flow), dim=-1)

        z = torch.cat([z_flow, z_poi], dim=-1)
        return z_flow, z_poi, z

    @staticmethod
    def _cross_view_ntxent(z_a: torch.Tensor, z_b: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Symmetric cross-view InfoNCE:
        - a->b and b->a averaged
        """
        logits_ab = (z_a @ z_b.t()) / tau
        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss_ab = F.cross_entropy(logits_ab, labels)
        loss_ba = F.cross_entropy(logits_ab.t(), labels)
        return 0.5 * (loss_ab + loss_ba)
