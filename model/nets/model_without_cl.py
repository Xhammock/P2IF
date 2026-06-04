from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nets.model_aug import UrbanModelAug


class UrbanModelAugWithoutCL(UrbanModelAug):
    """
    Ablation variant: W/o CL (removes spatially aware contrastive constraints).

    "Spatially aware" refers to the negative-sample masking in the main model's NT-Xent loss:
    - Spatial-graph neighbors are not treated as negatives
    - Strong OD neighbors are not treated as negatives

    This variant keeps view augmentation and contrastive learning but disables neighbor
    masking, equivalent to standard in-batch NT-Xent.
    """

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss = self._ntxent(z1, z2, self.tau, allow_mask=None)
        return loss, {"loss": loss.item()}
