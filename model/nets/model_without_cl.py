from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nets.model_aug import UrbanModelAug


class UrbanModelAugWithoutCL(UrbanModelAug):
    """
    消融实验变体：W/o CL（去除空间感知对比学习约束）。

    这里“空间感知”具体对应主模型 NT-Xent 中的负样本屏蔽策略：
    - 空间图邻居不作为负样本
    - OD 强邻居不作为负样本

    本变体保留视图增强与对比学习框架，但不做邻居屏蔽，等价于普通 in-batch NT-Xent。
    """

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss = self._ntxent(z1, z2, self.tau, allow_mask=None)
        return loss, {"loss": loss.item()}

