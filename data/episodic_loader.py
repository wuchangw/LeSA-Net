import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch


@dataclass
class QueryEpisode:
    query_image: torch.Tensor
    query_mask: torch.Tensor
    class_name: str
    dataset_name: str
    query_case_id: str


@dataclass
class CaseRecord:
    case_id: str
    pet_path: Path
    ct_path: Path
    label_path: Path
    positive_voxels: int


class DatasetRegistry:
    def __init__(
        self,
        data_root: str,
        split_json: str,
        images_dir: Optional[str] = None,
        labels_dir: Optional[str] = None,
        skip_negative_samples: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.images_dir = Path(images_dir) if images_dir else self.data_root / "imagesTr"
        self.labels_dir = Path(labels_dir) if labels_dir else self.data_root / "labelsTr"
        self.split_json = Path(split_json)
        self.skip_negative_samples = skip_negative_samples
        self.dataset_name = "psma_pet_ct"
        self.class_name = "lesion"
        self.splits: Dict[str, List[CaseRecord]] = {"train": [], "val": []}
        self.removed_missing: Dict[str, List[str]] = {"train": [], "val": []}
        self.removed_negative: Dict[str, List[str]] = {"train": [], "val": []}
        self._label_sum_cache: Dict[str, int] = {}
        self._build()

    def _build(self) -> None:
        split_data = self._load_split_json()
        for split_name in ("train", "val"):
            case_ids = split_data.get(split_name, [])
            for case_id in case_ids:
                record = self._make_case_record(case_id)
                if record is None:
                    self.removed_missing[split_name].append(case_id)
                    continue
                if self.skip_negative_samples and record.positive_voxels <= 0:
                    self.removed_negative[split_name].append(case_id)
                    continue
                self.splits[split_name].append(record)

        for split_name, records in self.splits.items():
            if len(records) < 2:
                raise RuntimeError(
                    f"Split '{split_name}' has only {len(records)} usable positive cases. "
                    "At least 2 are required for support retrieval."
                )

    def _load_split_json(self) -> Dict[str, List[str]]:
        payload = json.loads(self.split_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if not payload:
                raise RuntimeError(f"Split file is empty: {self.split_json}")
            payload = payload[0]
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected split file format: {self.split_json}")
        return payload

    def _make_case_record(self, case_id: str) -> Optional[CaseRecord]:
        pet_path = self.images_dir / f"{case_id}_0000.nii.gz"
        ct_path = self.images_dir / f"{case_id}_0001.nii.gz"
        label_path = self.labels_dir / f"{case_id}.nii.gz"
        if not (pet_path.exists() and ct_path.exists() and label_path.exists()):
            return None

        positive_voxels = self._get_positive_voxels(label_path)
        return CaseRecord(
            case_id=case_id,
            pet_path=pet_path,
            ct_path=ct_path,
            label_path=label_path,
            positive_voxels=positive_voxels,
        )

    def _get_positive_voxels(self, label_path: Path) -> int:
        cache_key = str(label_path)
        if cache_key not in self._label_sum_cache:
            label = nib.load(str(label_path)).get_fdata()
            self._label_sum_cache[cache_key] = int((label > 0).sum())
        return self._label_sum_cache[cache_key]

    def get_split_records(self, split_name: str) -> List[CaseRecord]:
        return list(self.splits[split_name])

    def describe(self) -> str:
        lines = [
            f"Dataset root: {self.data_root}",
            f"Images dir: {self.images_dir}",
            f"Labels dir: {self.labels_dir}",
            f"Split json: {self.split_json}",
        ]
        for split_name in ("train", "val"):
            lines.append(
                f"{split_name}: usable={len(self.splits[split_name])}, "
                f"missing_removed={len(self.removed_missing[split_name])}, "
                f"negative_removed={len(self.removed_negative[split_name])}"
            )
        return "\n".join(lines)


class EpisodicQueryLoader:
    def __init__(
        self,
        registry: DatasetRegistry,
        split_name: str,
        spatial_size: Tuple[int, int, int] = (32, 64, 64),
        augment: bool = True,
        query_limit: Optional[int] = None,
    ) -> None:
        self.registry = registry
        self.split_name = split_name
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.augment = augment
        self.records = registry.get_split_records(split_name)
        self.query_limit = query_limit
        self._deterministic_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        if self.query_limit is None:
            return len(self.records)
        return min(self.query_limit, len(self.records))

    def __iter__(self) -> Iterable[QueryEpisode]:
        records = list(self.records)
        if self.split_name == "train":
            random.shuffle(records)
        if self.query_limit is not None:
            records = records[: self.query_limit]
        for record in records:
            deterministic = self.split_name != "train"
            query_image, query_mask = self.load_case_record(
                record,
                deterministic=deterministic,
                augment=self.augment,
                use_cache=False,
            )
            yield QueryEpisode(
                query_image=query_image,
                query_mask=query_mask,
                class_name=self.registry.class_name,
                dataset_name=self.registry.dataset_name,
                query_case_id=record.case_id,
            )

    def sample_support_records(
        self,
        query_case_id: str,
        pool_size: Optional[int],
        shuffle: bool,
    ) -> List[CaseRecord]:
        candidates = [record for record in self.records if record.case_id != query_case_id]
        if pool_size is None or pool_size >= len(candidates):
            if shuffle:
                random.shuffle(candidates)
            return candidates
        if shuffle:
            return random.sample(candidates, pool_size)
        return candidates[:pool_size]

    def load_case_record(
        self,
        record: CaseRecord,
        deterministic: bool,
        augment: bool,
        use_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic and not augment and use_cache and record.case_id in self._deterministic_cache:
            image, mask = self._deterministic_cache[record.case_id]
            return image.clone(), mask.clone()

        pet = nib.load(str(record.pet_path)).get_fdata().astype(np.float32)
        ct = nib.load(str(record.ct_path)).get_fdata().astype(np.float32)
        label = (nib.load(str(record.label_path)).get_fdata() > 0).astype(np.float32)

        image = np.stack([pet, ct], axis=0)
        image, label = self._extract_foreground_patch(image, label, deterministic=deterministic)
        image = self._normalize_image(image)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(label[None, ...]).float()

        if augment:
            image_tensor, mask_tensor = self._apply_augmentation(image_tensor, mask_tensor)

        if deterministic and not augment and use_cache:
            self._deterministic_cache[record.case_id] = (image_tensor.clone(), mask_tensor.clone())

        return image_tensor, mask_tensor

    def _extract_foreground_patch(
        self,
        image: np.ndarray,
        label: np.ndarray,
        deterministic: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fg_coords = np.argwhere(label > 0)
        if fg_coords.size == 0:
            raise RuntimeError("Negative samples should have been filtered before loading.")

        if deterministic:
            center = np.round(fg_coords.mean(axis=0)).astype(int)
        else:
            center = fg_coords[random.randrange(len(fg_coords))]

        cropped_image = self._crop_or_pad(image, center)
        cropped_label = self._crop_or_pad(label[None, ...], center)[0]
        return cropped_image, cropped_label

    def _crop_or_pad(self, array: np.ndarray, center: np.ndarray) -> np.ndarray:
        _, depth, height, width = array.shape
        target_d, target_h, target_w = self.spatial_size
        output = np.zeros((array.shape[0], target_d, target_h, target_w), dtype=array.dtype)

        starts = [
            int(center[0] - target_d // 2),
            int(center[1] - target_h // 2),
            int(center[2] - target_w // 2),
        ]
        ends = [starts[0] + target_d, starts[1] + target_h, starts[2] + target_w]
        limits = [depth, height, width]

        src_slices = []
        dst_slices = []
        for start, end, limit in zip(starts, ends, limits):
            src_start = max(start, 0)
            src_end = min(end, limit)
            dst_start = max(-start, 0)
            dst_end = dst_start + (src_end - src_start)
            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))

        output[
            :,
            dst_slices[0],
            dst_slices[1],
            dst_slices[2],
        ] = array[
            :,
            src_slices[0],
            src_slices[1],
            src_slices[2],
        ]
        return output

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        normalized = image.copy()
        for channel in range(normalized.shape[0]):
            values = normalized[channel]
            foreground = values[np.abs(values) > 1e-6]
            if foreground.size == 0:
                mean = float(values.mean())
                std = float(values.std())
            else:
                mean = float(foreground.mean())
                std = float(foreground.std())
            std = max(std, 1e-6)
            normalized[channel] = (values - mean) / std
        return normalized

    def _apply_augmentation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-3])
            mask = torch.flip(mask, dims=[-3])
        return image.contiguous(), mask.contiguous()
