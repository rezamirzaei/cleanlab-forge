from __future__ import annotations

from typing import Any

import numpy as np

_VISION_DEPS_HELP = 'Install with: `pip install -e .` (or `uv sync`).'


def require_vision_deps() -> tuple[Any, Any]:
    """Import and return (torch, torchvision)."""
    try:
        import torch
        import torchvision
    except ImportError as e:
        raise RuntimeError(f"Missing optional vision dependencies.\n{_VISION_DEPS_HELP}") from e
    try:
        _ = torch.tensor([0]).numpy()
    except Exception as e:
        raise RuntimeError(
            "Torch is installed but NumPy integration is unavailable. "
            "Ensure you have `numpy<2` installed.\n"
            f"{_VISION_DEPS_HELP}"
        ) from e
    return torch, torchvision


def clip_boxes_xyxy(boxes: np.ndarray, *, w: int, h: int) -> np.ndarray:
    """Clip boxes to image boundaries."""
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(float)
    b = boxes.astype(float).copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(max(0, w - 1)))
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(max(0, h - 1)))
    x1 = np.minimum(b[:, 0], b[:, 2])
    x2 = np.maximum(b[:, 0], b[:, 2])
    y1 = np.minimum(b[:, 1], b[:, 3])
    y2 = np.maximum(b[:, 1], b[:, 3])
    b[:, 0], b[:, 2], b[:, 1], b[:, 3] = x1, x2, y1, y2
    b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1.0)
    b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1.0)
    return b


def corrupt_boxes(boxes: np.ndarray, *, img_w: int, img_h: int, seed: int) -> tuple[np.ndarray, bool]:
    """Corrupt boxes by dropping or shifting."""
    rng = np.random.default_rng(seed=seed)
    b = boxes.astype(float).copy()
    if len(b) == 0:
        return b.reshape(0, 4), False
    if rng.random() < 0.5:
        drop = int(rng.integers(0, len(b)))
        b = np.delete(b, drop, axis=0)
        return b.reshape(-1, 4), True

    dx = int(rng.integers(-25, 26))
    dy = int(rng.integers(-25, 26))
    b[:, [0, 2]] = b[:, [0, 2]] + dx
    b[:, [1, 3]] = b[:, [1, 3]] + dy
    b = clip_boxes_xyxy(b, w=img_w, h=img_h)
    return b.reshape(-1, 4), True


def corrupt_binary_mask(mask: np.ndarray, *, seed: int) -> tuple[np.ndarray, bool]:
    """Corrupt mask by zeroing out a region."""
    rng = np.random.default_rng(seed=seed)
    m = mask.astype(int).copy()
    if m.sum() == 0:
        return m, False
    h, w = m.shape
    rh = int(rng.integers(max(1, h // 10), max(2, h // 4)))
    rw = int(rng.integers(max(1, w // 10), max(2, w // 4)))
    y0 = int(rng.integers(0, max(1, h - rh)))
    x0 = int(rng.integers(0, max(1, w - rw)))
    m[y0 : y0 + rh, x0 : x0 + rw] = 0
    return m, True
