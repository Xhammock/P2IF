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
    Two-stage architecture: spatial GraphSAGE + OD cross-attention + projection head.
    Uses Region2Vec-style community-oriented loss.
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
        # Region2Vec-style loss parameters
        spatial_lambda: float = 0.1,  # spatial constraint weight (deprecated, kept for compatibility)
        loss_eps: float = 1e-15,  # epsilon in loss (consistent with Region2Vec)
        hops_threshold: int = 5,  # hop-distance threshold; hops below this are zeroed in the constraint
        hops_matrix_path: str | None = None,  # path to hop-distance matrix file (required)
        loss_type: str = "divreg",  # loss type: 'div' or 'divreg'
    ):
        super().__init__()
        self.dims = dims
        in_dim = dims["poi"] + dims["res"] + \
            dims["vis"] + dims.get("street", 0)

        # Region2Vec loss parameters
        self.spatial_lambda = spatial_lambda  # kept but unused
        self.loss_eps = loss_eps
        self.hops_threshold = hops_threshold
        self.hops_matrix_path = hops_matrix_path
        self.loss_type = loss_type

        # Require hop-distance matrix path
        if not self.hops_matrix_path:
            raise ValueError("hops_matrix_path is required and cannot be empty")

        # Cache spatial hop-distance weight matrix (avoid recomputation)
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
        # Map spatial output to Query / per-modality Key-Value dimensions
        self.q_proj = nn.Linear(hidden_dim, self.q_dim)
        self.poi_proj = nn.Linear(hidden_dim, self.poi_dim)
        self.vis_proj = nn.Linear(hidden_dim, self.vis_dim)
        self.street_proj = nn.Linear(
            hidden_dim, self.street_dim) if self.street_dim > 0 else None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # Single encode pass (no view augmentation)
        h_spatial, h_od, z = self._encode(batch)

        # Region2Vec-style loss
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
        Return spatial output, OD output, and L2-normalized projection z.
        For inference/visualization; caller should use no_grad.
        No data augmentation at inference time.
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

        # Q/K from spatial output: Q uses res subspace, K uses poi+vis+street (attention scores only)
        # Value uses full h_spatial
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
        Load spatial hop-distance weight matrix from file.

        Returns:
            hops_weight_matrix: [N, N] float tensor weight matrix
        """
        # Reuse cache if valid and on same device
        if (
            self._hops_weight_matrix is not None
            and self._hops_weight_matrix.device == device
            and self._hops_weight_matrix_num_nodes == num_nodes
        ):
            return self._hops_weight_matrix

        # Load hop-distance matrix from file
        if not os.path.exists(self.hops_matrix_path):
            raise FileNotFoundError(f"Hop-distance matrix file not found: {self.hops_matrix_path}")

        import numpy as np
        hops_m = np.loadtxt(self.hops_matrix_path, delimiter=',')
        assert hops_m.shape[0] == hops_m.shape[1] == num_nodes, \
            f"Hop-distance matrix size mismatch: expected {num_nodes}x{num_nodes}, got {hops_m.shape}"

        # Convert to weights: hops_m = 1/(log(hops_m + EPS) + 1)
        zero_entries = hops_m < self.hops_threshold
        hops_m_weighted = 1 / (np.log(hops_m + self.loss_eps) + 1)
        hops_m_weighted[zero_entries] = 0

        # Debug: hop-distance stats on first load only
        if self._hops_weight_matrix is None:
            num_valid_hops = (~zero_entries).sum()
            if num_valid_hops > 0:
                print(f"[Hop matrix] file={self.hops_matrix_path}, threshold={self.hops_threshold}, "
                      f"valid_pairs={num_valid_hops}/{num_nodes*num_nodes}, "
                      f"hop_range=[{hops_m[~zero_entries].min():.1f}, {hops_m[~zero_entries].max():.1f}], "
                      f"weight_range=[{hops_m_weighted[~zero_entries].min():.4f}, {hops_m_weighted[~zero_entries].max():.4f}]")
            else:
                print(f"[Warning] No valid node pairs in hop matrix (threshold={self.hops_threshold}); spatial constraint disabled")

        hops_weight_matrix = torch.FloatTensor(hops_m_weighted).to(device)

        # Cache result
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
        Following Region2Vec (SIGSPATIAL 2022).

        Args:
            z: [N, D] node embeddings (L2-normalized)
            g_od: OD graph with flow edge features
            device: compute device

        Returns:
            loss: total loss
            info: loss detail dict
        """
        num_nodes = z.size(0)

        # 1. Build flow matrix and masks from OD graph
        src, dst = g_od.edges()
        src = src.to(device)
        dst = dst.to(device)

        # Initialize flow matrix with raw flow values
        flow_matrix_raw = torch.zeros(
            (num_nodes, num_nodes), device=device, dtype=torch.float32)

        if src.numel() > 0:
            # Prefer raw flow if provided by the dataset
            if "flow_raw" in g_od.edata:
                flow_raw = g_od.edata["flow_raw"].to(device).view(-1).float()
            else:
                # Otherwise approximate inverse of normalization
                flow_normalized = g_od.edata["flow"].to(
                    device).view(-1).float()
                # Assume (log1p(x) - mean) / std; approximate with exp (may be inaccurate)
                flow_raw = torch.exp(
                    flow_normalized.clamp(min=-10, max=10)) - 1.0
                flow_raw = flow_raw.clamp(min=0.0)  # non-negative

            # Symmetrize: flow(i,j) = flow(j,i) = max(flow_ij, flow_ji)
            flow_matrix_raw[src, dst] = torch.maximum(
                flow_matrix_raw[src, dst],
                flow_raw
            )
            flow_matrix_raw[dst, src] = flow_matrix_raw[src, dst]

        # Positive/negative sample masks
        pos_mask = (flow_matrix_raw > 0).float()  # flow > 0
        neg_mask = (flow_matrix_raw == 0).float()  # flow == 0
        neg_mask.fill_diagonal_(0)  # exclude self

        # Flow weights: log(flow + EPS), consistent with Region2Vec
        flow_labels = torch.log(flow_matrix_raw + self.loss_eps)

        # Count positive/negative pairs
        N_pos = pos_mask.sum().item()
        N_neg = neg_mask.sum().item()

        # 2. Pairwise L2 distance between embeddings
        z_expanded_i = z.unsqueeze(1)  # [N, 1, D]
        z_expanded_j = z.unsqueeze(0)  # [1, N, D]
        pdist = torch.norm(z_expanded_i - z_expanded_j, dim=2, p=2)  # [N, N]

        # 3. Load hop-distance weight matrix from file
        hops_weight_matrix = self._load_hops_weight_matrix(num_nodes, device)

        # 4. Spatial constraint term
        loss_hops = torch.sum(pdist * hops_weight_matrix) + self.loss_eps
        num_hops_pairs = (hops_weight_matrix > 0).sum().item()

        # 5. Loss (Region2Vec div / divreg)
        if self.loss_type == "div":
            # div: loss = sum(pdist * labels * pos_mask) / (sum(pdist * neg_mask) + loss_hops)
            loss_train = torch.sum(pdist * flow_labels * pos_mask) / (
                torch.sum(pdist * neg_mask) + loss_hops
            )
        elif self.loss_type == "divreg":
            # divreg (normalized): loss = sum(pdist * labels * pos_mask) * N_neg / (N_pos * (sum(pdist * neg_mask) + loss_hops))
            if N_pos > 0 and N_neg > 0:
                loss_train = torch.sum(pdist * flow_labels * pos_mask) * N_neg / (
                    N_pos * (torch.sum(pdist * neg_mask) + loss_hops)
                )
            else:
                # Fallback when no positive or negative pairs
                loss_train = torch.sum(pdist * flow_labels * pos_mask) / (
                    torch.sum(pdist * neg_mask) + loss_hops + self.loss_eps
                )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Per-term loss values for logging
        loss_pos_value = torch.sum(
            pdist * flow_labels * pos_mask).item() if N_pos > 0 else 0.0
        loss_neg_value = torch.sum(
            pdist * neg_mask).item() if N_neg > 0 else 0.0

        info = {
            "loss": loss_train.item(),
            # positive term: sum(pdist * log(flow) * pos_mask)
            "loss_pos": loss_pos_value,
            "loss_neg": loss_neg_value,  # negative term: sum(pdist * neg_mask)
            "loss_hops": loss_hops.item(),  # spatial: sum(pdist * hops_weight_matrix)
            "loss_spatial": loss_hops.item(),  # alias for training scripts
            "num_pos_pairs": int(N_pos),  # pairs with flow > 0
            "num_neg_pairs": int(N_neg),  # pairs with flow == 0
            # pairs in spatial constraint (hop distance >= threshold)
            "num_hops_pairs": int(num_hops_pairs),
            "hops_weight_sum": hops_weight_matrix.sum().item(),  # sum of hop weight matrix
        }

        return loss_train, info
