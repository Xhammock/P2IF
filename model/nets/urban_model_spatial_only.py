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
    Spatial-adjacency-only GraphSAGE (ablation):
    - Spatial GraphSAGE only, no OD layer
    - View-augmented contrastive learning (spatially aware NT-Xent)
    - Optional toggles for poi / res / vis / street
    - Outputs z (projected representation) only
    """

    def __init__(
        self,
        dims: Dict[str, int],
        hidden_dim: int = 256,
        sage_layers: int = 1,
        dropout: float = 0.1,
        proj_dim: int = 128,
        feat_drop_ratio: float = 0.1,
        edge_drop_ratio: float = 0.075,
        noise_std: float = 0.01,
        tau: float = 0.1,
        use_poi: bool = True,
        use_res: bool = True,
        use_vis: bool = True,
        use_street: bool = True,
    ):
        super().__init__()
        self.dims = dims

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
            raise ValueError("At least one modality (poi/res/vis/street) must be enabled")

        self.original_poi_dim = dims["poi"]
        self.original_res_dim = dims["res"]
        self.original_vis_dim = dims["vis"]
        self.original_street_dim = dims.get("street", 0)

        self.tau = tau
        self.feat_drop_ratio = feat_drop_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.noise_std = noise_std

        self._spatial_allow_mask: torch.Tensor | None = None
        self._spatial_allow_mask_num_nodes: int | None = None
        self._spatial_allow_mask_num_edges: int | None = None

        self.spatial = SpatialSAGE(
            in_dim, hidden_dim, num_layers=sage_layers, dropout=dropout)

        self.ffn = FeedForward(hidden_dim, dropout)
        self.proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Training: two augmented views and contrastive loss."""
        view1_spatial, view1_feat = self._create_view1(
            batch["g_spatial"], batch["feat"])
        view2_spatial, view2_feat = self._create_view2(
            batch["g_spatial"], batch["feat"])

        batch_v1 = {"g_spatial": view1_spatial, "feat": view1_feat}
        batch_v2 = {"g_spatial": view2_spatial, "feat": view2_feat}
        z_v1 = self._encode(batch_v1)
        z_v2 = self._encode(batch_v2)

        self._ensure_spatial_allow_mask(
            batch["g_spatial"], device=z_v1.device)

        loss, info = self._contrastive_loss(z_v1, z_v2)

        info.update({"z": z_v1})
        return loss, info

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return L2-normalized z for inference/visualization (no augmentation)."""
        return self._encode(batch)

    def _extract_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Slice full feature vector [poi, res, vis, street] for ablation."""
        parts = []
        start_idx = 0

        if self.use_poi:
            parts.append(feats[:, start_idx:start_idx + self.original_poi_dim])
        start_idx += self.original_poi_dim

        if self.use_res:
            parts.append(feats[:, start_idx:start_idx + self.original_res_dim])
        start_idx += self.original_res_dim

        if self.use_vis:
            parts.append(feats[:, start_idx:start_idx + self.original_vis_dim])
        start_idx += self.original_vis_dim

        if self.use_street and self.original_street_dim > 0:
            parts.append(feats[:, start_idx:start_idx + self.original_street_dim])
        if self.original_street_dim > 0:
            start_idx += self.original_street_dim

        return torch.cat(parts, dim=-1) if len(parts) > 0 else feats

    def _encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Spatial GraphSAGE -> FFN -> projection -> L2 normalize."""
        g_spatial = batch["g_spatial"]
        feats = batch["feat"]

        feats = self._extract_features(feats)

        h_spatial = self.spatial(g_spatial, feats)
        h_spatial = self.ffn(h_spatial)

        return F.normalize(self.proj(h_spatial), dim=-1)

    def _create_view1(self, g_spatial, feats: torch.Tensor) -> Tuple:
        """View 1: feature dropout + Gaussian noise."""
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
        return view1_spatial, view1_feat

    def _create_view2(self, g_spatial, feats: torch.Tensor) -> Tuple:
        """View 2: drop-edge + Gaussian noise."""
        view2_feat = feats
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(view2_feat) * self.noise_std
            view2_feat = view2_feat + noise

        view2_spatial = self._drop_edges(g_spatial, self.edge_drop_ratio)
        view2_spatial.ndata["feat"] = view2_feat.clone()
        return view2_spatial, view2_feat

    def _drop_edges(self, g, drop_ratio: float):
        """Randomly drop a fraction of edges."""
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
        Build spatial allow_mask for NT-Xent.
        allow_mask[i, j] == True means j may be a negative for i.
        Diagonal forced True for positive pairs.
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

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """View-augmented contrastive loss with in-batch negatives."""
        loss = self._ntxent(z1, z2, self.tau, allow_mask=self._spatial_allow_mask)
        return loss, {"loss": loss.item()}

    def _ntxent(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        tau: float,
        allow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard vectorized NT-Xent loss (in-batch negatives)."""
        logits = torch.matmul(z1, z2.t()) / tau

        if allow_mask is not None:
            logits = logits.masked_fill(~allow_mask, float("-inf"))

        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)
