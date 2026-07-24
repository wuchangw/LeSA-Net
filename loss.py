import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as connected_components

from utils.losses import CombinedLoss as BaseCombinedLoss


class SizeAwareBCELoss(nn.Module):
    def __init__(
        self,
        voxel_weight_scale=0.5,
        size_reference=128.0,
        component_gamma=0.5,
        max_component_weight=3.0,
        min_component_voxels=4,
    ):
        super().__init__()
        self.voxel_weight_scale = voxel_weight_scale
        self.size_reference = size_reference
        self.component_gamma = component_gamma
        self.max_component_weight = max_component_weight
        self.min_component_voxels = min_component_voxels

    def _build_weight_map(self, target: torch.Tensor) -> torch.Tensor:
        target_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
        weight_map = torch.ones_like(target, dtype=torch.float32)
        if target_np.sum() == 0:
            return weight_map

        labeled_components, num_components = connected_components(
            target_np,
            structure=np.ones((3, 3, 3), dtype=np.uint8),
        )
        if num_components == 0:
            return weight_map

        positive_mask = target > 0.5
        positive_weights = torch.ones_like(target, dtype=torch.float32)

        for component_index in range(1, num_components + 1):
            component_mask_np = labeled_components == component_index
            voxel_count = int(component_mask_np.sum())
            if voxel_count < self.min_component_voxels:
                continue

            size_ratio = max(self.size_reference / max(float(voxel_count), 1.0), 1.0)
            component_weight = min(size_ratio ** self.component_gamma, self.max_component_weight)
            component_mask = torch.from_numpy(component_mask_np).to(device=target.device, dtype=torch.bool)
            positive_weights[component_mask] = component_weight

        positive_values = positive_weights[positive_mask]
        if positive_values.numel() == 0:
            return weight_map

        positive_values = positive_values / positive_values.mean().clamp_min(1e-6)
        weight_map[positive_mask] = 1.0 + self.voxel_weight_scale * (positive_values - 1.0)
        return weight_map

    def forward(self, predictions, targets):
        if predictions.shape[1] != 1:
            raise ValueError("SizeAwareBCELoss currently supports binary segmentation only.")

        if targets.dim() == 5:
            targets = targets.squeeze(1)

        targets = targets.float()
        logits = predictions.squeeze(1)
        per_voxel_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        sample_losses = []
        for batch_index in range(logits.shape[0]):
            weight_map = self._build_weight_map(targets[batch_index])
            sample_losses.append((per_voxel_bce[batch_index] * weight_map).mean())

        return torch.stack(sample_losses).mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        dice_weight=0.5,
        ce_weight=0.5,
        size_aware_weight=0.1,
        size_aware_start_weight=None,
        size_aware_ramp_start_epoch=1,
        size_aware_ramp_end_epoch=30,
        voxel_weight_scale=0.5,
        size_reference=128.0,
        component_gamma=0.5,
        max_component_weight=3.0,
        min_component_voxels=4,
        smooth=1e-5,
        ignore_index=None,
    ):
        super().__init__()
        self.base_loss = BaseCombinedLoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            smooth=smooth,
            ignore_index=ignore_index,
        )
        self.target_size_aware_weight = float(size_aware_weight)
        if size_aware_start_weight is None:
            size_aware_start_weight = size_aware_weight
        self.initial_size_aware_weight = float(size_aware_start_weight)
        self.size_aware_weight = float(size_aware_start_weight)
        self.size_aware_ramp_start_epoch = int(size_aware_ramp_start_epoch)
        self.size_aware_ramp_end_epoch = int(size_aware_ramp_end_epoch)
        self.size_aware_bce = SizeAwareBCELoss(
            voxel_weight_scale=voxel_weight_scale,
            size_reference=size_reference,
            component_gamma=component_gamma,
            max_component_weight=max_component_weight,
            min_component_voxels=min_component_voxels,
        )

    def set_size_aware_weight(self, weight: float) -> None:
        self.size_aware_weight = float(weight)

    def get_size_aware_weight(self) -> float:
        return float(self.size_aware_weight)

    def get_ramp_weight(self, epoch: int) -> float:
        if self.size_aware_ramp_end_epoch <= self.size_aware_ramp_start_epoch:
            return self.target_size_aware_weight
        if epoch <= self.size_aware_ramp_start_epoch:
            return self.initial_size_aware_weight
        if epoch >= self.size_aware_ramp_end_epoch:
            return self.target_size_aware_weight

        progress = (epoch - self.size_aware_ramp_start_epoch) / max(
            self.size_aware_ramp_end_epoch - self.size_aware_ramp_start_epoch,
            1,
        )
        return float(self.initial_size_aware_weight + progress * (self.target_size_aware_weight - self.initial_size_aware_weight))

    def forward(self, predictions, targets):
        base_total_loss, dice_loss, ce_loss = self.base_loss(predictions, targets)
        size_aware_loss = self.size_aware_bce(predictions, targets)
        total_loss = base_total_loss + self.size_aware_weight * size_aware_loss
        auxiliary_loss = ce_loss + self.size_aware_weight * size_aware_loss
        return total_loss, dice_loss, auxiliary_loss
