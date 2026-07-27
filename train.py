import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml

from data.episodic_loader import DatasetRegistry, EpisodicQueryLoader
from loss import CombinedLoss
from models.lesa_model import LeSANet
from training.episodic_trainer import EpisodicTrainer


DEFAULT_CONFIG = {
    "data_root": "/data/cyf/shared_data/mri-pet/CT_PET/DeepPSMA_v1",
    "images_dir": "/data/cyf/shared_data/mri-pet/CT_PET/DeepPSMA_v1/imagesTr",
    "labels_dir": "/data/cyf/shared_data/mri-pet/CT_PET/DeepPSMA_v1/labelsTr",
    "split_json": "/data/cyf/shared_data/mri-pet/CT_PET/DeepPSMA_v1/psma_split.json",
    "checkpoint_path": "best_lesanet.pth",
    "log_path": "log_lesanet.txt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs": 100,
    "query_limit": None,
    "support_pool_size": 16,
    "top_k_supports": 2,
    "spatial_size": [32, 64, 64],
    "skip_negative_samples": True,
    "in_channels": 2,
    "base_channels": 8,
    "num_classes": 1,
    "fusion_temperature": 1.0,
    "retrieval_grid_size": [8, 8, 8],
    "retrieval_percentile": 85.0,
    "optimizer": {
        "type": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-5,
    },
    "scheduler": {
        "type": "CosineAnnealingLR",
        "T_max": 50,
        "eta_min": 1e-6,
    },
    "loss": {
        "dice_weight": 0.5,
        "ce_weight": 0.5,
        "size_aware_weight": 0.05,
        "size_aware_start_weight": 0.0,
        "size_aware_ramp_start_epoch": 1,
        "size_aware_ramp_end_epoch": 30,
        "voxel_weight_scale": 0.3,
        "size_reference": 128.0,
        "component_gamma": 0.5,
        "max_component_weight": 2.0,
        "min_component_voxels": 8,
        "smooth": 1e-5,
    },
}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def ensure_utf8_console():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def deep_update(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str):
    config = dict(DEFAULT_CONFIG)
    if config_path and Path(config_path).exists():
        override = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        config = deep_update(config, override)
    return config


def build_registry(config):
    registry = DatasetRegistry(
        data_root=config["data_root"],
        images_dir=config["images_dir"],
        labels_dir=config["labels_dir"],
        split_json=config["split_json"],
        skip_negative_samples=config.get("skip_negative_samples", True),
    )
    print(registry.describe())
    for split_name in ("train", "val"):
        missing = registry.removed_missing[split_name]
        if missing:
            print(f"{split_name} missing cases removed: {', '.join(missing)}")
        negatives = registry.removed_negative[split_name]
        if negatives:
            print(f"{split_name} negative cases removed: {', '.join(negatives)}")
    return registry


def create_model(config):
    model = LeSANet(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        num_classes=config["num_classes"],
        fusion_temperature=config["fusion_temperature"],
    )
    info = model.get_model_info()
    print(f"Model parameters: {info['total_parameters']:,}")
    return model


def create_optimizer(model, config):
    optimizer_cfg = config["optimizer"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg.get("lr", 1e-4),
        weight_decay=optimizer_cfg.get("weight_decay", 1e-5),
    )
    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "CosineAnnealingLR")
    if scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", config["num_epochs"]),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
        )
    else:
        scheduler = None
    return optimizer, scheduler


def run_training(config, resume=None):
    registry = build_registry(config)
    spatial_size = tuple(config["spatial_size"])
    train_loader = EpisodicQueryLoader(
        registry=registry,
        split_name="train",
        spatial_size=spatial_size,
        augment=True,
        query_limit=config.get("query_limit"),
    )
    val_loader = EpisodicQueryLoader(
        registry=registry,
        split_name="val",
        spatial_size=spatial_size,
        augment=False,
        query_limit=None,
    )

    model = create_model(config)
    optimizer, scheduler = create_optimizer(model, config)
    loss_fn = CombinedLoss(**config["loss"])
    trainer = EpisodicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=config["device"],
        support_pool_size=config["support_pool_size"],
        top_k_supports=config["top_k_supports"],
        retrieval_grid_size=tuple(config["retrieval_grid_size"]),
        retrieval_percentile=config["retrieval_percentile"],
    )
    if resume:
        trainer.load_checkpoint(resume)
    trainer.train(
        num_epochs=config["num_epochs"],
        checkpoint_path=config["checkpoint_path"],
        config=config,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train LeSANet on a PET/CT lesion segmentation dataset.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root.")
    parser.add_argument("--images-dir", type=str, default=None, help="imagesTr path.")
    parser.add_argument("--labels-dir", type=str, default=None, help="labelsTr path.")
    parser.add_argument("--split-json", type=str, default=None, help="Split json path.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Best checkpoint path.")
    parser.add_argument("--log-path", type=str, default=None, help="Single training log file.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--query-limit", type=int, default=None, help="Optional limit on train queries per epoch.")
    parser.add_argument("--support-pool-size", type=int, default=None, help="Number of random supports in the support pool.")
    parser.add_argument("--top-k-supports", type=int, default=None, help="Number of retrieved supports passed into the model.")
    return parser.parse_args()


def main():
    ensure_utf8_console()
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        "data_root": args.data_root,
        "images_dir": args.images_dir,
        "labels_dir": args.labels_dir,
        "split_json": args.split_json,
        "checkpoint_path": args.checkpoint_path,
        "log_path": args.log_path,
        "device": args.device,
        "num_epochs": args.num_epochs,
        "query_limit": args.query_limit,
        "support_pool_size": args.support_pool_size,
        "top_k_supports": args.top_k_supports,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    log_path = Path(config["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8-sig") as log_file:
        tee = TeeStream(sys.__stdout__, log_file)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            print("LeSANet training")
            print("=" * 60)
            print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            run_training(config, resume=args.resume)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()
