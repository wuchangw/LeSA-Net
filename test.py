import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from data.episodic_loader import DatasetRegistry, EpisodicQueryLoader
from evaluation.metrics import CaseMetrics, compute_case_metrics, format_case_table, format_summary_block, summarize_case_metrics
from loss import CombinedLoss
from models.lesa_model import LeSANet
from train import DEFAULT_CONFIG


def ensure_utf8_console():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def compute_pet_descriptor(images: torch.Tensor, percentile: float, grid_size):
    pet = torch.relu(images[:, 0:1])
    flat = pet.flatten(start_dim=1)
    threshold = torch.quantile(
        flat.detach(),
        percentile / 100.0,
        dim=1,
    ).view(-1, 1, 1, 1, 1)
    hotspot = pet * (pet >= threshold).float()
    empty = hotspot.flatten(start_dim=1).sum(dim=1) <= 1e-6
    if empty.any():
        hotspot[empty] = pet[empty]
    pooled = F.interpolate(
        hotspot,
        size=grid_size,
        mode="trilinear",
        align_corners=False,
    )
    descriptor = pooled.flatten(start_dim=1)
    return F.normalize(descriptor, p=2, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a trained LeSANet checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="best_lesanet.pth", help="Checkpoint path.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_CONFIG["data_root"], help="Dataset root.")
    parser.add_argument("--images-dir", type=str, default=DEFAULT_CONFIG["images_dir"], help="imagesTr path.")
    parser.add_argument("--labels-dir", type=str, default=DEFAULT_CONFIG["labels_dir"], help="labelsTr path.")
    parser.add_argument("--split-json", type=str, default=DEFAULT_CONFIG["split_json"], help="Split json path.")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"], help="cuda or cpu.")
    parser.add_argument("--output", type=str, default="test_lesanet.txt", help="Metrics report output.")
    return parser.parse_args()


def create_eval_loss(loss_cfg):
    if "size_aware_weight" in loss_cfg:
        return CombinedLoss(**loss_cfg)
    return CombinedLoss(
        dice_weight=loss_cfg.get("dice_weight", 0.5),
        ce_weight=loss_cfg.get("ce_weight", 0.5),
        size_aware_weight=loss_cfg.get("size_aware_weight", 0.1),
        voxel_weight_scale=loss_cfg.get("voxel_weight_scale", 0.5),
        size_reference=loss_cfg.get("size_reference", 128.0),
        component_gamma=loss_cfg.get("component_gamma", 0.5),
        max_component_weight=loss_cfg.get("max_component_weight", 3.0),
        min_component_voxels=loss_cfg.get("min_component_voxels", 4),
        smooth=loss_cfg.get("smooth", 1e-5),
    )


def build_model_from_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_config = checkpoint.get("run_config", DEFAULT_CONFIG)
    model = LeSANet(
        in_channels=run_config["in_channels"],
        base_channels=run_config["base_channels"],
        num_classes=run_config["num_classes"],
        fusion_temperature=run_config.get("fusion_temperature", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, run_config


def main():
    ensure_utf8_console()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, run_config = build_model_from_checkpoint(str(checkpoint_path), args.device)
    registry = DatasetRegistry(
        data_root=args.data_root,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        split_json=args.split_json,
        skip_negative_samples=True,
    )
    loader = EpisodicQueryLoader(
        registry=registry,
        split_name="val",
        spatial_size=tuple(run_config["spatial_size"]),
        augment=False,
        query_limit=None,
    )
    loss_cfg = run_config["loss"]
    loss_fn = create_eval_loss(loss_cfg)
    support_pool_size = run_config["support_pool_size"]
    top_k_supports = run_config["top_k_supports"]
    retrieval_percentile = run_config.get("retrieval_percentile", 85.0)
    retrieval_grid_size = tuple(run_config.get("retrieval_grid_size", [8, 8, 8]))

    case_rows = []
    print("LeSANet validation")
    print("=" * 60)
    print(registry.describe())
    with torch.no_grad():
        for step, episode in enumerate(loader, start=1):
            query_image = episode.query_image.unsqueeze(0).to(args.device)
            query_mask = episode.query_mask.unsqueeze(0).to(args.device)
            support_records = loader.sample_support_records(
                query_case_id=episode.query_case_id,
                pool_size=support_pool_size,
                shuffle=False,
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

            support_images = torch.stack(support_images, dim=0).to(args.device)
            support_masks = torch.stack(support_masks, dim=0).to(args.device)
            query_descriptor = compute_pet_descriptor(query_image, retrieval_percentile, retrieval_grid_size)
            support_descriptors = compute_pet_descriptor(support_images, retrieval_percentile, retrieval_grid_size)
            scores = torch.matmul(query_descriptor, support_descriptors.transpose(0, 1)).squeeze(0)
            top_k = min(top_k_supports, support_images.shape[0])
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=0)
            top_support_images = support_images[top_indices]
            top_support_masks = support_masks[top_indices]
            top_support_case_ids = [support_case_ids[int(index)] for index in top_indices.tolist()]

            outputs = model.forward_multi_support(
                query_image=query_image,
                support_images=top_support_images,
                support_masks=top_support_masks,
                support_scores=top_scores,
                support_case_ids=top_support_case_ids,
                return_details=True,
            )
            prediction = outputs["logits"]
            total_loss, _, _ = loss_fn(prediction, query_mask.squeeze(1))
            probabilities = torch.sigmoid(prediction)
            prediction_mask = (probabilities > 0.5).float()
            metrics = compute_case_metrics(
                prediction_mask,
                query_mask,
            )
            case_rows.append(
                CaseMetrics(
                    case_id=episode.query_case_id,
                    reference_case_id=outputs["best_support_case_id"],
                    metrics=metrics,
                    status="OK",
                )
            )
            print(
                f"val step {step:03d}/{len(loader):03d} "
                f"loss={total_loss.item():.4f} dice={metrics['Dice']:.4f} "
                f"ref={outputs['best_support_case_id']} qry={episode.query_case_id}"
            )

    summary = summarize_case_metrics(case_rows)
    summary_text = format_summary_block(summary)
    table_text = format_case_table(case_rows)
    report = f"{summary_text}\n{table_text}\n"
    Path(args.output).write_text(report, encoding="utf-8-sig")
    print(summary_text)
    print(table_text)


if __name__ == "__main__":
    main()
