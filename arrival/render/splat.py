"""Depth composite of a static room view and moving simulator geometry.

The caller must render both layers from the same camera pose and resolution.
This module performs the final occlusion decision; it does not train a Gaussian
splat or extract moving geometry from a full-scene MuJoCo render.
"""

from __future__ import annotations

import math

import numpy as np

DEPTH_MARGIN = 0.01


class SplatCompositeNotImplemented(NotImplementedError):
    """A rendered static and dynamic layer was not supplied."""


def render(
    static_rgb: np.ndarray | None = None,
    static_depth: np.ndarray | None = None,
    dynamic_rgb: np.ndarray | None = None,
    dynamic_depth: np.ndarray | None = None,
    *,
    margin: float = DEPTH_MARGIN,
) -> np.ndarray:
    """Return RGB with a dynamic pixel where it is > margin metres nearer.

    Depth is optical-axis distance in metres. Zero, negative, NaN or infinity
    means no surface. When both layers lack depth, the static RGB is retained.
    The result is a new array; inputs are never modified.
    """
    if any(layer is None for layer in (static_rgb, static_depth, dynamic_rgb, dynamic_depth)):
        raise SplatCompositeNotImplemented("static RGB/depth and moving-geometry RGB/depth are required")
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite, nonnegative distance in metres")
    static_rgb, dynamic_rgb = np.asarray(static_rgb), np.asarray(dynamic_rgb)
    static_depth, dynamic_depth = np.asarray(static_depth), np.asarray(dynamic_depth)
    if static_rgb.ndim != 3 or static_rgb.shape[2] != 3 or static_rgb.dtype != np.uint8:
        raise ValueError("static_rgb must be HxWx3 uint8")
    if dynamic_rgb.shape != static_rgb.shape or dynamic_rgb.dtype != np.uint8:
        raise ValueError("dynamic_rgb must match static_rgb's HxWx3 uint8 shape")
    size = static_rgb.shape[:2]
    if (
        static_depth.shape != size
        or dynamic_depth.shape != size
        or static_depth.dtype.kind not in "fi"
        or dynamic_depth.dtype.kind not in "fi"
    ):
        raise ValueError("depth layers must be numeric HxW arrays matching RGB")
    static_ok = np.isfinite(static_depth) & (static_depth > 0)
    dynamic_ok = np.isfinite(dynamic_depth) & (dynamic_depth > 0)
    closer = dynamic_ok & (~static_ok | (dynamic_depth < static_depth - margin))
    return np.where(closer[..., None], dynamic_rgb, static_rgb)
