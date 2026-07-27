from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, label as connected_components


METRIC_ORDER = ["Dice", "IoU", "PPV", "Sensitivity", "HD95", "ASSD", "MLQS"]
CASE_SIZE_ORDER = ["GT Vol", "Pred Vol", "GT Area", "Pred Area"]


@dataclass
class CaseMetrics:
    case_id: str
    reference_case_id: str
    metrics: Dict[str, float]
    status: str


def select_reference_query_pairs(records: Sequence) -> List[Tuple[object, object]]:
    if len(records) < 2:
        raise RuntimeError("At least 2 positive cases are required for evaluation.")
    pairs = []
    total = len(records)
    for query_idx, query_record in enumerate(records):
        reference_idx = (query_idx + 1) % total
        if reference_idx == query_idx:
            reference_idx = (reference_idx + 1) % total
        pairs.append((records[reference_idx], query_record))
    return pairs


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _extract_surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3,) * max(mask.ndim, 1), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _surface_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.sum() == 0 or target.sum() == 0:
        return np.array([], dtype=np.float32)
    target_surface = _extract_surface(target)
    source_surface = _extract_surface(source)
    if target_surface.sum() == 0 or source_surface.sum() == 0:
        return np.array([], dtype=np.float32)
    target_distance = distance_transform_edt(~target_surface)
    return target_distance[source_surface]


def _surface_area(mask: np.ndarray) -> float:
    mask_int = np.asarray(mask, dtype=np.int8)
    if mask_int.size == 0:
        return 0.0
    padded = np.pad(mask_int, 1, mode="constant")
    area = 0.0
    for axis in range(mask_int.ndim):
        area += float(np.abs(np.diff(padded, axis=axis)).sum())
    return area


def _connected_component_masks(mask: np.ndarray) -> List[np.ndarray]:
    structure = np.ones((3,) * max(mask.ndim, 1), dtype=np.uint8)
    labeled_components, num_components = connected_components(mask.astype(np.uint8), structure=structure)
    return [(labeled_components == component_index) for component_index in range(1, num_components + 1)]


def _harmonic_mean(x: float, y: float) -> float:
    if x + y <= 0:
        return 0.0
    return (2.0 * x * y) / (x + y)


def _macro_lesion_quality_score(prediction: np.ndarray, target: np.ndarray) -> float:
    gt_components = _connected_component_masks(target)
    if not gt_components:
        return 1.0 if prediction.sum() == 0 else 0.0

    pred_components = _connected_component_masks(prediction)
    pred_component_map = np.zeros_like(prediction, dtype=np.int32)
    for component_index, component_mask in enumerate(pred_components, start=1):
        pred_component_map[component_mask] = component_index

    lesion_scores = []
    for gt_component in gt_components:
        overlapping_labels = np.unique(pred_component_map[gt_component])
        overlapping_labels = overlapping_labels[overlapping_labels > 0]
        if overlapping_labels.size == 0:
            lesion_scores.append(0.0)
            continue

        matched_prediction = np.isin(pred_component_map, overlapping_labels)
        intersection = float(np.logical_and(matched_prediction, gt_component).sum())
        gt_volume = float(gt_component.sum())
        pred_volume = float(matched_prediction.sum())
        lesion_dice = _safe_divide(2.0 * intersection, pred_volume + gt_volume)
        lesion_recall = _safe_divide(intersection, gt_volume)
        lesion_scores.append(_harmonic_mean(lesion_dice, lesion_recall))

    return float(np.mean(lesion_scores)) if lesion_scores else 0.0


def compute_case_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    include_lesion_metrics: bool = False,
) -> Dict[str, float]:
    del include_lesion_metrics
    pred = prediction.detach().cpu().numpy().astype(bool).squeeze()
    gt = target.detach().cpu().numpy().astype(bool).squeeze()

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    pred_vol = float(pred.sum())
    gt_vol = float(gt.sum())
    union = float(np.logical_or(pred, gt).sum())

    dice = 1.0 if pred_vol + gt_vol == 0 else _safe_divide(2.0 * tp, pred_vol + gt_vol)
    iou = 1.0 if union == 0 else _safe_divide(tp, union)
    ppv = 1.0 if pred_vol == 0 and gt_vol == 0 else _safe_divide(tp, tp + fp)
    sensitivity = 1.0 if pred_vol == 0 and gt_vol == 0 else _safe_divide(tp, tp + fn)

    dist_pred_to_gt = _surface_distances(pred, gt)
    dist_gt_to_pred = _surface_distances(gt, pred)
    all_surface_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred]) if (
        dist_pred_to_gt.size or dist_gt_to_pred.size
    ) else np.array([], dtype=np.float32)

    if pred_vol == 0 and gt_vol == 0:
        hd95 = 0.0
        assd = 0.0
    elif all_surface_distances.size == 0:
        hd95 = float("nan")
        assd = float("nan")
    else:
        hd95 = float(np.percentile(all_surface_distances, 95))
        assd = float(all_surface_distances.mean())

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "PPV": float(ppv),
        "Sensitivity": float(sensitivity),
        "HD95": float(hd95),
        "ASSD": float(assd),
        "MLQS": float(_macro_lesion_quality_score(pred, gt)),
        "GT Vol": gt_vol,
        "Pred Vol": pred_vol,
        "GT Area": float(_surface_area(gt)),
        "Pred Area": float(_surface_area(pred)),
    }


def summarize_case_metrics(
    case_metrics: Sequence[CaseMetrics],
    include_lesion_metrics: bool = False,
) -> Dict[str, Tuple[float, float]]:
    del include_lesion_metrics
    summary = {}
    for metric_name in METRIC_ORDER:
        values = np.array([item.metrics[metric_name] for item in case_metrics], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary[metric_name] = (float("nan"), float("nan"))
        else:
            summary[metric_name] = (float(finite_values.mean()), float(finite_values.std()))
    return summary


def format_summary_block(
    summary: Dict[str, Tuple[float, float]],
    include_lesion_metrics: bool = False,
) -> str:
    del include_lesion_metrics
    lines = ["=============================="]
    for metric_name in METRIC_ORDER:
        mean_value, std_value = summary[metric_name]
        lines.append(f"{metric_name:<12}: {mean_value:.4f} +/- {std_value:.4f}")
    return "\n".join(lines)


def format_case_table(
    case_metrics: Sequence[CaseMetrics],
    include_lesion_metrics: bool = False,
) -> str:
    del include_lesion_metrics
    header = (
        f"{'Case ID':<30} | {'Dice':>6} | {'IoU':>6} | {'PPV':>6} | {'Sen':>6} | "
        f"{'HD95':>8} | {'ASSD':>8} | {'MLQS':>6} | {'GT Vol':>8} | {'Pred Vol':>8} | "
        f"{'GT Area':>9} | {'Pred Area':>9} | Status"
    )
    lines = [
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for item in case_metrics:
        lines.append(
            f"{item.case_id:<30} | "
            f"{item.metrics['Dice']:>6.4f} | "
            f"{item.metrics['IoU']:>6.4f} | "
            f"{item.metrics['PPV']:>6.4f} | "
            f"{item.metrics['Sensitivity']:>6.4f} | "
            f"{item.metrics['HD95']:>8.4f} | "
            f"{item.metrics['ASSD']:>8.4f} | "
            f"{item.metrics['MLQS']:>6.4f} | "
            f"{int(round(item.metrics['GT Vol'])):>8} | "
            f"{int(round(item.metrics['Pred Vol'])):>8} | "
            f"{item.metrics['GT Area']:>9.2f} | "
            f"{item.metrics['Pred Area']:>9.2f} | "
            f"{item.status:>6}"
        )
    return "\n".join(lines)
