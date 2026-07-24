import os
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from evaluation.psma_metrics import CaseMetrics, compute_case_metrics, format_summary_block, summarize_case_metrics
from utils.losses import CombinedLoss, dice_score


class EpisodicTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cuda",
        support_pool_size: Optional[int] = None,
        top_k_supports: int = 2,
        retrieval_grid_size: Tuple[int, int, int] = (8, 8, 8),
        retrieval_percentile: float = 85.0,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = scheduler
        self.loss_fn = loss_fn or CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        self.device = device
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_dice = float("-inf")
        self.support_pool_size = support_pool_size
        self.top_k_supports = top_k_supports
        self.retrieval_grid_size = retrieval_grid_size
        self.retrieval_percentile = retrieval_percentile

    def _update_loss_schedule(self) -> Optional[float]:
        if not hasattr(self.loss_fn, "get_ramp_weight") or not hasattr(self.loss_fn, "set_size_aware_weight"):
            return None
        current_weight = self.loss_fn.get_ramp_weight(self.current_epoch)
        self.loss_fn.set_size_aware_weight(current_weight)
        return float(current_weight)

    def _build_support_pool(self, loader, query_case_id: str, shuffle: bool):
        support_records = loader.sample_support_records(
            query_case_id=query_case_id,
            pool_size=self.support_pool_size,
            shuffle=shuffle,
        )
        support_images = []
        support_masks = []
        support_case_ids = []
        for record in support_records:
            image, mask = loader.load_case_record(
                record,
                deterministic=True,
                augment=False,
                use_cache=True,
            )
            support_images.append(image)
            support_masks.append(mask)
            support_case_ids.append(record.case_id)
        support_images = torch.stack(support_images, dim=0).to(self.device)
        support_masks = torch.stack(support_masks, dim=0).to(self.device)
        return support_images, support_masks, support_case_ids

    def _compute_pet_descriptor(self, images: torch.Tensor) -> torch.Tensor:
        pet = torch.relu(images[:, 0:1])
        flat = pet.flatten(start_dim=1)
        threshold = torch.quantile(
            flat.detach(),
            self.retrieval_percentile / 100.0,
            dim=1,
        ).view(-1, 1, 1, 1, 1)
        hotspot = pet * (pet >= threshold).float()
        empty = hotspot.flatten(start_dim=1).sum(dim=1) <= 1e-6
        if empty.any():
            hotspot[empty] = pet[empty]
        pooled = F.interpolate(
            hotspot,
            size=self.retrieval_grid_size,
            mode="trilinear",
            align_corners=False,
        )
        descriptor = pooled.flatten(start_dim=1)
        return F.normalize(descriptor, p=2, dim=1)

    def _select_top_supports(
        self,
        query_image: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        support_case_ids,
    ):
        query_descriptor = self._compute_pet_descriptor(query_image)
        support_descriptors = self._compute_pet_descriptor(support_images)
        scores = torch.matmul(query_descriptor, support_descriptors.transpose(0, 1)).squeeze(0)
        top_k = min(self.top_k_supports, support_images.shape[0])
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=0)
        top_support_images = support_images[top_indices]
        top_support_masks = support_masks[top_indices]
        top_support_case_ids = [support_case_ids[int(index)] for index in top_indices.tolist()]
        return top_support_images, top_support_masks, top_support_case_ids, top_scores

    def episodic_training_step(self, episode) -> Dict[str, float]:
        query_image = episode.query_image.unsqueeze(0).to(self.device)
        query_mask = episode.query_mask.unsqueeze(0).to(self.device)
        support_images, support_masks, support_case_ids = self._build_support_pool(
            self.train_loader,
            query_case_id=episode.query_case_id,
            shuffle=True,
        )
        top_support_images, top_support_masks, top_support_case_ids, top_scores = self._select_top_supports(
            query_image,
            support_images,
            support_masks,
            support_case_ids,
        )

        self.optimizer.zero_grad()
        outputs = self.model.forward_multi_support(
            query_image=query_image,
            support_images=top_support_images,
            support_masks=top_support_masks,
            support_scores=top_scores,
            support_case_ids=top_support_case_ids,
            return_details=True,
        )
        prediction = outputs["logits"]
        total_loss, dice_loss_value, ce_loss_value = self.loss_fn(prediction, query_mask.squeeze(1))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            dice_metric = dice_score(torch.sigmoid(prediction), query_mask)

        return {
            "total_loss": float(total_loss.item()),
            "dice_loss": float(dice_loss_value.item()),
            "ce_loss": float(ce_loss_value.item()),
            "dice_score": float(dice_metric.item()),
            "class_name": episode.class_name,
            "dataset_name": episode.dataset_name,
            "query_case_id": episode.query_case_id,
            "reference_case_id": outputs["best_support_case_id"],
        }

    def validation_step(self, episode) -> Dict[str, float]:
        with torch.no_grad():
            query_image = episode.query_image.unsqueeze(0).to(self.device)
            query_mask = episode.query_mask.unsqueeze(0).to(self.device)
            support_images, support_masks, support_case_ids = self._build_support_pool(
                self.val_loader,
                query_case_id=episode.query_case_id,
                shuffle=False,
            )
            top_support_images, top_support_masks, top_support_case_ids, top_scores = self._select_top_supports(
                query_image,
                support_images,
                support_masks,
                support_case_ids,
            )
            outputs = self.model.forward_multi_support(
                query_image=query_image,
                support_images=top_support_images,
                support_masks=top_support_masks,
                support_scores=top_scores,
                support_case_ids=top_support_case_ids,
                return_details=True,
            )
            prediction = outputs["logits"]
            total_loss, dice_loss_value, ce_loss_value = self.loss_fn(prediction, query_mask.squeeze(1))
            prediction_mask = (torch.sigmoid(prediction) > 0.5).float()
            case_metrics = compute_case_metrics(
                prediction_mask,
                query_mask,
            )

        return {
            "val_total_loss": float(total_loss.item()),
            "val_dice_loss": float(dice_loss_value.item()),
            "val_ce_loss": float(ce_loss_value.item()),
            "val_dice_score": float(case_metrics["Dice"]),
            "case_metrics": case_metrics,
            "class_name": episode.class_name,
            "dataset_name": episode.dataset_name,
            "query_case_id": episode.query_case_id,
            "reference_case_id": outputs["best_support_case_id"],
        }

    def _run_epoch(self, loader, train: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        epoch_metrics = defaultdict(list)
        class_metrics = defaultdict(list)
        case_metric_rows = []
        start_time = time.time()

        for episode in loader:
            metrics = self.episodic_training_step(episode) if train else self.validation_step(episode)
            key_prefix = "" if train else "val_"
            dice_key = f"{key_prefix}dice_score"
            loss_key = f"{key_prefix}total_loss"

            for key, value in metrics.items():
                if key not in {"class_name", "dataset_name", "query_case_id", "reference_case_id", "case_metrics"}:
                    epoch_metrics[key].append(value)

            class_metrics[metrics["class_name"]].append(metrics[dice_key])

            if not train:
                case_metric_rows.append(
                    CaseMetrics(
                        case_id=metrics["query_case_id"],
                        reference_case_id=metrics["reference_case_id"],
                        metrics=metrics["case_metrics"],
                        status="OK",
                    )
                )

            if train:
                self.global_step += 1

        averaged = {key: float(np.mean(values)) for key, values in epoch_metrics.items()}
        averaged["epoch_time"] = time.time() - start_time
        per_class = {key: float(np.mean(values)) for key, values in class_metrics.items()}
        print(
            f"{'Train' if train else 'Val'} summary: "
            f"loss={averaged[loss_key]:.4f} "
            f"dice={averaged[dice_key]:.4f} "
            f"time={averaged['epoch_time']:.1f}s"
        )

        if not train:
            averaged["summary_metrics"] = summarize_case_metrics(case_metric_rows)
            averaged["summary_text"] = format_summary_block(averaged["summary_metrics"])
            averaged["case_metrics"] = case_metric_rows

        return averaged, per_class

    def train_epoch(self):
        return self._run_epoch(self.train_loader, train=True)

    def validate_epoch(self):
        if self.val_loader is None:
            return None
        return self._run_epoch(self.val_loader, train=False)

    def save_checkpoint(self, filepath: str, config: Dict) -> None:
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "best_val_dice": self.best_val_dice,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": getattr(self.model, "config", {}),
                "run_config": config,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_dice = float(checkpoint.get("best_val_dice", float("-inf")))
        print(f"Loaded checkpoint from {filepath}")

    def train(self, num_epochs: int, checkpoint_path: str, config: Dict) -> None:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            scheduled_weight = self._update_loss_schedule()
            print("")
            print(f"Epoch {self.current_epoch}/{num_epochs}")
            print("-" * 60)
            if scheduled_weight is not None:
                print(f"size_aware_weight={scheduled_weight:.4f}")
            train_metrics, _ = self.train_epoch()
            val_result = self.validate_epoch()
            current_score = train_metrics["dice_score"]

            if val_result is not None:
                val_metrics, _ = val_result
                current_score = val_metrics["val_dice_score"]
                print(val_metrics["summary_text"])

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_score)
                else:
                    self.scheduler.step()

            if current_score >= self.best_val_dice:
                self.best_val_dice = current_score
                self.save_checkpoint(checkpoint_path, config=config)
                print(f"Updated best checkpoint: {checkpoint_path} (dice={current_score:.4f})")

        print("")
        print(f"Training finished. Best dice={self.best_val_dice:.4f}")
