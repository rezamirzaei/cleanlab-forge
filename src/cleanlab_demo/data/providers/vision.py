"""
Vision Data Providers.

Provides data providers for vision tasks (detection, segmentation).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from cleanlab_demo.tasks.vision.provider import VisionDataProvider
from cleanlab_demo.utils.download import download_file


class PennFudanPedProvider(VisionDataProvider):
    """
    Penn-Fudan Pedestrian dataset provider.

    Object detection and instance segmentation of pedestrians.
    Source: PennFudanPed dataset used in torchvision detection tutorials.
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "torchvision.datasets.PennFudanPed"

    @property
    def model_name(self) -> str:
        return "maskrcnn_resnet50_fpn (COCO)"

    @property
    def class_mapping(self) -> dict[str, int]:
        return {"dataset_person": 0, "coco_person": 1}

    def _ensure_dataset(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)

        ds_dir = root / "PennFudanPed"
        images_dir = ds_dir / "PNGImages"
        masks_dir = ds_dir / "PedMasks"
        if images_dir.exists() and masks_dir.exists() and any(images_dir.iterdir()):
            return ds_dir

        zip_path = root / "PennFudanPed.zip"
        urls = [
            "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
            "https://download.pytorch.org/tutorial/PennFudanPed.zip",
        ]
        last_err: Exception | None = None
        for url in urls:
            try:
                download_file(url, zip_path)
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise RuntimeError(f"Failed to download PennFudanPed dataset: {last_err}") from last_err

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

        if not images_dir.exists() or not masks_dir.exists():
            raise RuntimeError(
                f"PennFudanPed extraction failed: missing `{images_dir}` or `{masks_dir}`"
            )
        return ds_dir

    def load(
        self, data_dir: Path, max_images: int, seed: int, **kwargs: Any
    ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
        """Load PennFudanPed images and ground truth."""
        try:
            import torchvision  # noqa: F401
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "vision deps required. Install with: `pip install -e .` (or `uv sync`)."
            ) from e

        ds_dir = self._ensure_dataset(data_dir)
        images_dir = ds_dir / "PNGImages"
        masks_dir = ds_dir / "PedMasks"
        to_tensor = torchvision.transforms.functional.to_tensor

        img_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
        if not img_paths:
            raise RuntimeError(f"PennFudanPed dataset is empty: `{images_dir}`")

        rng = np.random.default_rng(seed=seed)
        n = min(int(max_images), len(img_paths))
        indices = sorted(rng.choice(len(img_paths), size=n, replace=False).astype(int).tolist())

        images: list[Any] = []
        gt_boxes: list[np.ndarray] = []
        gt_masks: list[np.ndarray] = []
        img_sizes: list[tuple[int, int]] = []

        for idx in indices:
            img_path = img_paths[idx]
            mask_path = masks_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                raise RuntimeError(f"Missing mask for `{img_path.name}`: expected `{mask_path.name}`")

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)

            img_w, img_h = img.size
            img_sizes.append((img_w, img_h))
            images.append(to_tensor(img))

            m = np.asarray(mask)
            if m.ndim == 3:
                # Some PNG decoders may expand palette PNGs to RGB; fall back to first channel.
                m = m[:, :, 0]

            obj_ids = np.unique(m)
            obj_ids = obj_ids[obj_ids != 0]

            boxes_list: list[list[float]] = []
            for obj_id in obj_ids.tolist():
                obj_mask = m == obj_id
                if not np.any(obj_mask):
                    continue
                ys, xs = np.where(obj_mask)
                xmin = float(xs.min())
                xmax = float(xs.max())
                ymin = float(ys.min())
                ymax = float(ys.max())
                boxes_list.append([xmin, ymin, xmax, ymax])

            gt_boxes.append(np.asarray(boxes_list, dtype=float).reshape(-1, 4))
            gt_masks.append((m > 0).astype(int))

        return images, gt_boxes, gt_masks, img_sizes

    def get_model(self) -> tuple[Any, int]:
        """Get MaskRCNN model and COCO person class ID."""
        try:
            import torchvision
        except ImportError as e:
            raise RuntimeError(
                "vision deps required. Install with: `pip install -e .` (or `uv sync`)."
            ) from e

        det = torchvision.models.detection
        try:
            weights = det.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            model = det.maskrcnn_resnet50_fpn(weights=weights)
        except Exception:
            try:
                model = det.maskrcnn_resnet50_fpn(pretrained=True)
            except Exception:
                model = det.maskrcnn_resnet50_fpn()

        return model, 1  # COCO person class ID
