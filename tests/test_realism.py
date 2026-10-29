"""EVAL-1a: the twin scored against its own render is exact, a moved camera scores worse, invalid depth is not
counted, training frames are refused, and a bundle that blocks the realism claim reports null."""

import copy

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.eval.realism import evaluate_frames, look_at, perturb_pose, render_twin, score, side_by_side_png  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

SIZE = (64, 80)  # height, width
K = np.array([[70.0, 0, 40.0], [0, 70.0, 32.0], [0, 0, 1.0]])


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


@pytest.fixture(scope="module")
def pose(twin):
    xs = [p for w in twin["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in twin["walls"] for p in (w["start"][1], w["end"][1])]
    return look_at((max(xs) - 0.4, max(ys) - 0.4, 1.5), ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.8))


@pytest.fixture(scope="module")
def ref(twin, pose):
    return render_twin(twin, K, pose, SIZE)


def _frame(ref, pose, fid="h0"):
    rgb, depth = ref
    return {"id": fid, "rgb": rgb, "depth": depth, "valid_depth_mask": depth > 0, "K": K, "cam_to_room": pose}


def test_render_shape_and_depth_in_metres(ref):
    rgb, depth = ref
    assert rgb.shape == (*SIZE, 3) and rgb.dtype == np.uint8 and depth.shape == SIZE
    hit = depth[depth > 0]
    assert hit.size > 0.5 * depth.size and 0.3 < hit.min() and hit.max() < 15.0  # a room, not clip-plane values
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 5  # not a blank frame


def test_robot_is_not_rendered(twin, monkeypatch):
    base = (1.3, 1.0, 0.75)
    t = look_at((1.3, 1.0, 1.3), (1.3, 1.0, 0.0))  # straight down at the robot's base
    centre = (SIZE[0] // 2, SIZE[1] // 2)
    _, depth = render_twin(twin, K, t, SIZE, base_pos=base)
    assert abs(depth[centre] - 1.3) < 0.05  # the floor, not a pedestal or link
    monkeypatch.setattr("arrival.eval.realism._hidden_bodies", lambda m: [])  # control: unhidden, the robot is seen
    _, shown = render_twin(twin, K, t, SIZE, base_pos=base)
    assert shown[centre] < depth[centre] - 0.2


def test_self_render_scores_zero(twin, pose, ref):
    rgb, depth = ref
    rgb2, depth2 = render_twin(twin, K, pose, SIZE)
    s = score(rgb, depth, rgb2, depth2, depth > 0)
    assert s["depth_l1_m"] < 1e-3 and s["rgb_mae"] == 0 and s["rgb_one_minus_ssim"] < 1e-9
    assert (
        s["rgb_psnr_db"] is None
        and s["n_depth"] == int((depth > 0).sum())
        and s["n_rgb"] == rgb.shape[0] * rgb.shape[1]
    )


def test_shifted_and_rotated_camera_scores_worse(twin, pose, ref):
    rgb, depth = ref
    good = score(rgb, depth, *ref, depth > 0)
    rgb2, depth2 = render_twin(twin, K, perturb_pose(pose, 0.15, 5.0), SIZE)
    bad = score(rgb, depth, rgb2, depth2, depth > 0)
    assert bad["depth_l1_m"] > good["depth_l1_m"] + 1e-3
    assert bad["rgb_mae"] > good["rgb_mae"] and bad["rgb_one_minus_ssim"] > good["rgb_one_minus_ssim"]
    assert bad["rgb_psnr_db"] is not None


def test_invalid_depth_and_robot_pixels_are_excluded(ref):
    rgb, depth = ref
    valid = depth > 0
    base = score(rgb, depth, rgb, depth, valid)
    holes = valid.copy()
    holes[:20] = False
    s = score(rgb, depth, rgb, depth, holes)
    assert s["n_depth"] == int(holes.sum()) < base["n_depth"]
    wrong = depth.copy()
    wrong[:20] += 5.0  # a large error on the invalid rows must not reach the mean
    assert score(rgb, wrong, rgb, depth, holes)["depth_l1_m"] == 0
    robot = np.zeros(SIZE, bool)
    robot[:, :40] = True
    s = score(rgb, depth, rgb, depth, valid, robot_mask=robot)
    assert s["n_rgb"] == SIZE[0] * 40 and s["n_depth"] == int((valid & ~robot).sum())
    assert s["rgb_mask_coverage"] == 0.5
    s = score(rgb, depth, rgb, depth, np.zeros(SIZE, bool))
    assert s["n_depth"] == 0 and s["depth_l1_m"] is None


def test_sim_no_hit_pixels_are_counted_not_hidden(ref):
    rgb, depth = ref
    sim = depth.copy()
    sim[:10] = 0.0
    s = score(rgb, depth, rgb, sim, depth > 0)
    assert s["depth_sim_no_hit"] == int((depth[:10] > 0).sum())


def test_evaluate_frames_refuses_non_heldout_ids(twin, pose, ref):
    with pytest.raises(ValueError, match="train_3"):
        evaluate_frames(twin, [_frame(ref, pose, "h0"), _frame(ref, pose, "train_3")], ["h0", "h1"])
    r = evaluate_frames(twin, [_frame(ref, pose, "h0")], ["h0", "h1"])["realism"]
    assert r["frames"] == 1 and r["heldout_only"] and r["depth_l1_m"] < 1e-3 and r["rgb_mae"] == 0


def test_blocked_realism_returns_null_with_reason(twin, pose, ref):
    t = copy.deepcopy(twin)
    t["blocks_claims"] = ["geometry", "realism"]
    out = evaluate_frames(t, [_frame(ref, pose, "h0")], ["h0"])
    assert out["realism"] is None and "realism" in out["reason"]
    with pytest.raises(ValueError):  # the held-out check still comes first
        evaluate_frames(t, [_frame(ref, pose, "other")], ["h0"])


def test_off_centre_intrinsics_are_refused(twin, pose):
    k = K.copy()
    k[0, 2] = 60.0
    with pytest.raises(ValueError, match="principal point"):
        render_twin(twin, k, pose, SIZE)


def test_side_by_side_png(tmp_path, ref):
    pytest.importorskip("PIL")
    rgb, depth = ref
    p = tmp_path / "cmp.png"
    side_by_side_png(p, rgb, rgb, depth, depth)
    from PIL import Image

    assert Image.open(p).size == (SIZE[1] * 4, SIZE[0])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("up_sign", [-1, 1])
def test_look_at_parallel_up_produces_valid_pose(axis, direction, up_sign):
    eye = np.array([1.0, 2.0, 3.0])
    forward = np.eye(3)[axis] * direction
    with np.errstate(divide="raise", invalid="raise"):
        pose = look_at(eye, eye + forward, up=forward * up_sign)
    assert np.isfinite(pose).all()
    rotation = pose[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation[:, 2], forward)
    np.testing.assert_allclose(pose[:3, 3], eye)
    np.testing.assert_array_equal(pose[3], [0, 0, 0, 1])
