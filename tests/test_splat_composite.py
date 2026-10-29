import numpy as np
import pytest

from arrival.render.splat import DEPTH_MARGIN, render


def test_depth_composite_keeps_static_when_motion_is_behind_or_within_margin():
    room = np.full((1, 4, 3), 20, np.uint8)
    moving = np.full((1, 4, 3), 200, np.uint8)
    static_depth = np.array([[1.0, 1.0, 1.0, np.nan]], np.float32)
    moving_depth = np.array([[0.98, 0.995, 1.1, 0.5]], np.float32)
    before = room.copy()
    out = render(room, static_depth, moving, moving_depth)
    assert out[0, :, 0].tolist() == [200, 20, 20, 200]
    assert np.array_equal(room, before) and not np.shares_memory(out, room)
    assert DEPTH_MARGIN == 0.01


def test_invalid_dynamic_depth_does_not_cover_room():
    room = np.full((1, 4, 3), 20, np.uint8)
    moving = np.full((1, 4, 3), 200, np.uint8)
    out = render(room, np.ones((1, 4), np.float32), moving, np.array([[0, -1, np.nan, np.inf]], np.float32))
    assert np.array_equal(out, room)


@pytest.mark.parametrize("kwargs", [{"margin": -0.1}, {"margin": float("nan")}, {"margin": True}])
def test_bad_depth_margin_rejected(kwargs):
    rgb = np.zeros((2, 2, 3), np.uint8)
    depth = np.ones((2, 2), np.float32)
    with pytest.raises(ValueError, match="margin"):
        render(rgb, depth, rgb, depth, **kwargs)


def test_mismatched_camera_layers_rejected():
    rgb = np.zeros((2, 2, 3), np.uint8)
    depth = np.ones((2, 2), np.float32)
    with pytest.raises(ValueError, match="dynamic_rgb"):
        render(rgb, depth, np.zeros((3, 2, 3), np.uint8), depth)
    with pytest.raises(ValueError, match="depth layers"):
        render(rgb, depth, rgb, np.ones((3, 2), np.float32))
