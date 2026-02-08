from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cleanlab_demo.tasks.vision.provider import VisionDataProvider
from cleanlab_demo.tasks.vision.schemas import (
    ObjectDetectionSummary,
    SegmentationSummary,
    VisionDetectionSegmentationConfig,
    VisionDetectionSegmentationResult,
    VisionNotes,
)
from cleanlab_demo.tasks.vision.utils import corrupt_binary_mask, corrupt_boxes, require_vision_deps


class VisionDetectionSegmentationTask:
    """Object detection + instance segmentation label issue detection with Cleanlab."""

    def __init__(self, data_provider: VisionDataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: VisionDetectionSegmentationConfig) -> VisionDetectionSegmentationResult:
        torch, _ = require_vision_deps()

        from cleanlab.object_detection.filter import find_label_issues as find_od_issues
        from cleanlab.segmentation.rank import get_label_quality_scores as seg_quality_scores

        data_dir = config.data_dir or Path("./data")
        images, gt_boxes, gt_masks, img_sizes = self.data_provider.load(
            data_dir, config.max_images, config.seed
        )
        n = len(images)

        model, target_class_id = self.data_provider.get_model()
        model.eval()

        with torch.inference_mode():
            outputs = model(images)

        rng = np.random.default_rng(seed=config.seed)

        # ---------------------------------------------------------------------
        # Object Detection
        # ---------------------------------------------------------------------
        corrupt_od_set: set[int] = set(
            rng.choice(n, size=round(config.corrupt_frac * n), replace=False).astype(int).tolist()
        )

        labels_od_noisy: list[dict[str, Any]] = []
        predictions_od: list[list[np.ndarray]] = []

        for i, (out, boxes) in enumerate(zip(outputs, gt_boxes, strict=True)):
            img_w, img_h = img_sizes[i]

            noisy_boxes, did_corrupt = (
                corrupt_boxes(boxes, img_w=img_w, img_h=img_h, seed=config.seed + i)
                if i in corrupt_od_set
                else (boxes.astype(float), False)
            )
            if i in corrupt_od_set and not did_corrupt:
                corrupt_od_set.discard(i)

            labels_od_noisy.append(
                {
                    "bboxes": noisy_boxes.astype(float),
                    "labels": np.zeros(len(noisy_boxes), dtype=int),
                    "image_name": f"{i}",
                }
            )

            out_boxes = out["boxes"].cpu().numpy().astype(float)
            out_labels = out["labels"].cpu().numpy().astype(int)
            out_scores = out["scores"].cpu().numpy().astype(float)
            keep = (out_labels == target_class_id) & (out_scores >= float(config.score_threshold))
            b = out_boxes[keep]
            s = out_scores[keep]
            pred = (
                np.concatenate([b, s[:, None]], axis=1) if len(b) else np.zeros((0, 5), dtype=float)
            )
            predictions_od.append([pred])

        od_issue_mask = find_od_issues(labels=labels_od_noisy, predictions=predictions_od)
        od_flagged = set(np.flatnonzero(np.asarray(od_issue_mask, dtype=bool)).astype(int).tolist())
        od_tp = len(od_flagged & corrupt_od_set)
        od_recall = float(od_tp / len(corrupt_od_set)) if corrupt_od_set else 0.0
        od_precision = float(od_tp / len(od_flagged)) if od_flagged else 0.0

        # ---------------------------------------------------------------------
        # Segmentation
        # ---------------------------------------------------------------------
        corrupt_seg_set: set[int] = set(
            rng.choice(n, size=round(config.corrupt_frac * n), replace=False).astype(int).tolist()
        )

        gt_masks_noisy: list[np.ndarray] = []
        for i, m in enumerate(gt_masks):
            m2, did = (
                corrupt_binary_mask(m, seed=config.seed + 10_000 + i)
                if i in corrupt_seg_set
                else (m, False)
            )
            if i in corrupt_seg_set and not did:
                corrupt_seg_set.discard(i)
            gt_masks_noisy.append(m2.astype(int))

        pred_probs_by_img: list[np.ndarray] = []
        max_h = max(m.shape[0] for m in gt_masks_noisy)
        max_w = max(m.shape[1] for m in gt_masks_noisy)

        for out, (img_w, img_h) in zip(outputs, img_sizes, strict=True):
            out_labels = out["labels"].cpu().numpy().astype(int)
            out_scores = out["scores"].cpu().numpy().astype(float)
            out_masks = out.get("masks")
            if out_masks is None:
                p_target = np.zeros((img_h, img_w), dtype=float)
            else:
                keep = (out_labels == target_class_id) & (out_scores >= float(config.score_threshold))
                m = out_masks[keep].cpu().numpy()
                if len(m) == 0:
                    p_target = np.zeros((img_h, img_w), dtype=float)
                else:
                    p_target = m[:, 0].max(axis=0).astype(float)
            p_target = np.clip(p_target, 0.0, 1.0)
            pred_probs_by_img.append(np.stack([1.0 - p_target, p_target], axis=0))

        labels_arr = np.zeros((n, max_h, max_w), dtype=int)
        pred_probs_arr = np.zeros((n, 2, max_h, max_w), dtype=float)
        pred_probs_arr[:, 0, :, :] = 1.0

        for i, (m, probs) in enumerate(zip(gt_masks_noisy, pred_probs_by_img, strict=True)):
            h, w = m.shape
            labels_arr[i, :h, :w] = m
            pred_probs_arr[i, :, :h, :w] = probs

        seg_scores, _ = seg_quality_scores(labels_arr, pred_probs_arr, method="softmin", verbose=False)
        seg_scores = np.asarray(seg_scores, dtype=float)

        k = max(1, len(corrupt_seg_set))
        worst = set(np.argsort(seg_scores)[:k].astype(int).tolist())
        seg_tp = len(worst & corrupt_seg_set)
        seg_recall_at_k = float(seg_tp / len(corrupt_seg_set)) if corrupt_seg_set else 0.0
        seg_precision_at_k = float(seg_tp / len(worst)) if worst else 0.0

        return VisionDetectionSegmentationResult(
            dataset=self.data_provider.name,
            n_images=n,
            notes=VisionNotes(
                model=self.data_provider.model_name,
                class_mapping=self.data_provider.class_mapping,
                corrupt_frac=float(config.corrupt_frac),
            ),
            object_detection=ObjectDetectionSummary(
                score_threshold=float(config.score_threshold),
                n_corrupted_images=len(corrupt_od_set),
                n_flagged_images=len(od_flagged),
                precision_vs_corruption=float(od_precision),
                recall_vs_corruption=float(od_recall),
            ),
            segmentation=SegmentationSummary(
                score_threshold=float(config.score_threshold),
                n_corrupted_images=len(corrupt_seg_set),
                k_flagged=int(k),
                precision_at_k=float(seg_precision_at_k),
                recall_at_k=float(seg_recall_at_k),
            ),
        )


def run_vision_detection_segmentation(
    data_provider: VisionDataProvider,
    config: VisionDetectionSegmentationConfig | None = None,
    **kwargs: Any,
) -> VisionDetectionSegmentationResult:
    cfg = config or VisionDetectionSegmentationConfig(**kwargs)
    return VisionDetectionSegmentationTask(data_provider).run(cfg)

