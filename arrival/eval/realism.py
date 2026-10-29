"""EVAL-1a: how far the twin's render is from a real RGB-D frame taken at the same camera pose.

The metrics (`score`, `evaluate_frames`) are pure arrays in, numbers out. Camera convention is OpenCV: x right, y down, z forward, `cam_to_room` a 4x4 that maps camera coordinates to the twin's
z-up room frame, K a 3x3 pinhole matrix in pixels of the frame's own size. Depth is metres along the optical axis.

Metrics, all on pixels outside the robot mask:
  depth_l1_m          mean |depth_ref - depth_sim| over pixels where the reference depth is valid and the twin hit
                      something (n_depth of them)
  rgb_mae             mean absolute error of RGB scaled to [0, 1], over every non-robot pixel (n_rgb)
  rgb_one_minus_ssim  1 - mean SSIM (7x7 uniform window, luma, data range 1) over the same pixels
  rgb_psnr_db         PSNR from the RGB mean squared error; None when the images are identical
The twin renders the fixed room only: the robot and people are hidden, so a reference frame that shows them needs
a robot_mask (and the walkers get no mask here, so score frames without people).

`run_realism` (`python -m arrival realism <scan>`) drives the metrics above over a scan's frame manifest (arrival/scan/rgbd.py). It adds only what a
measurement needs to be quotable:

  * held-out frames only. Frames the manifest marks `train` are never scored; the JSON lists the ids it used and
    how many it left out. A scan with no held-out frame is refused.
  * invalid depth is masked: a pixel counts for depth only if the reference depth is finite and above 0 and its
    ARKit confidence (when the bundle has a confidence map) is at least --min-confidence (2 = high). The twin render
    hides the robot and people, and the recorder films the empty room, so no robot mask is needed or applied.
  * a misaligned-pose control: the same frames are scored again with the camera moved 0.15 m sideways and turned
    5 degrees. If depth error does not rise, the poses or the twin's frame are wrong and the number must not be
    quoted; the command then exits 1 and says so.
  * a four-photo or RoomPlan-only scan has no frames: it is refused, never scored. A bundle whose blocks_claims holds
    `realism` reports realism null with the reason.

The twin is a schematic box-and-colour render (MuJoCo, no textures), so rgb_mae and SSIM mostly measure missing
texture and lighting; depth_l1_m measures geometry. The JSON says so (`scope`). Frames larger than --max-side are
downscaled (intrinsics scaled with them, depth by nearest neighbour so no depth is invented by interpolation).

`measure` runs the same comparison as the worker's optional stage for an upload that carried RGB-D frames.
"""

from __future__ import annotations

import contextlib
import json
import math
import traceback
from pathlib import Path

import numpy as np

from arrival.provenance import git_state, sha256_file
from arrival.scan import rgbd
from arrival.sim.scene import build_mjcf

SSIM_WIN = 7
BASE = (0.0, 0.0, 0.75)  # the robot is hidden, so where it stands only has to be a valid pose
HIDDEN_GROUP = 4  # geom group of the hidden bodies; the scene option below switches it off
_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # MuJoCo cameras look down -z with y up


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """cam_to_room (OpenCV axes) for a camera at eye looking at target."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    if abs(float(z @ up)) > 1 - 1e-6:  # looking straight along up: any horizontal axis will do
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(z @ up)) > 1 - 1e-6:
            up = np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def perturb_pose(cam_to_room: np.ndarray, shift_m: float, yaw_deg: float) -> np.ndarray:
    """cam_to_room moved shift_m along the camera's own x axis and turned yaw_deg about its own y axis."""
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    T = np.array(cam_to_room, float)
    T[:3, 3] += T[:3, 0] * shift_m
    T[:3, :3] = T[:3, :3] @ R
    return T


def _hidden_bodies(model) -> list[int]:
    import mujoco

    out = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name == "robot_base" or name.startswith("person_"):
            out.append(b)
    return out


def render_twin(twin: dict, K, cam_to_room, size: tuple[int, int], base_pos=BASE) -> tuple[np.ndarray, np.ndarray]:
    """(rgb uint8 HxWx3, depth float32 HxW in metres) of the twin from the pose, robot and people excluded.
    size is (height, width); base_pos only says where the hidden robot stands. Depth is 0 where the camera sees no surface (sky above the open room). The
    principal point must sit within 2% of the image centre and fx within 2% of fy: the MuJoCo camera is symmetric."""
    import mujoco

    h, w = size
    K = np.asarray(K, float)
    T = np.asarray(cam_to_room, float)
    if K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError("K must be 3x3 and cam_to_room 4x4")
    if abs(K[0, 2] - w / 2) > 0.02 * w or abs(K[1, 2] - h / 2) > 0.02 * h or abs(K[0, 0] - K[1, 1]) > 0.02 * K[1, 1]:
        raise ValueError("render_twin needs a centred principal point and square pixels")
    model = mujoco.MjModel.from_xml_string(build_mjcf(twin, tuple(base_pos), n_people=0))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    hidden = set(_hidden_bodies(model))
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        while b > 0 and b not in hidden:
            b = int(model.body_parentid[b])
        if b in hidden:
            model.geom_group[g] = HIDDEN_GROUP
    cam = model.camera("overview")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (T[:3, :3] @ _CV_TO_MJ).ravel())
    model.cam_pos[cam.id] = T[:3, 3]
    model.cam_quat[cam.id] = quat
    model.cam_fovy[cam.id] = math.degrees(2 * math.atan(h / 2 / K[1, 1]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[HIDDEN_GROUP] = 0
    r = mujoco.Renderer(model, h, w)
    try:
        r.update_scene(data, "overview", scene_option=opt)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, "overview", scene_option=opt)
        depth = r.render().astype(np.float32)
    finally:
        r.close()
    far = float(model.stat.extent * model.vis.map.zfar)
    depth[~(depth < far * 0.99)] = 0.0
    return rgb, depth


def _ssim_map(a: np.ndarray, b: np.ndarray, win: int = SSIM_WIN) -> np.ndarray:
    """Per-pixel SSIM of two HxW float images in [0, 1], uniform window, edges padded by replication."""
    from numpy.lib.stride_tricks import sliding_window_view

    def mean(x):
        p = np.pad(x, win // 2, mode="edge")
        return sliding_window_view(p, (win, win)).mean(axis=(-1, -2))

    c1, c2 = 0.01**2, 0.03**2
    ma, mb = mean(a), mean(b)
    va, vb, cab = mean(a * a) - ma * ma, mean(b * b) - mb * mb, mean(a * b) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def score(rgb_ref, depth_ref, rgb_sim, depth_sim, valid_depth_mask, robot_mask=None) -> dict:
    """Errors between a reference frame and the twin render (see the module doc for each key)."""
    rgb_ref, rgb_sim = np.asarray(rgb_ref), np.asarray(rgb_sim)
    depth_ref, depth_sim = np.asarray(depth_ref, float), np.asarray(depth_sim, float)
    if rgb_ref.shape != rgb_sim.shape or depth_ref.shape != depth_sim.shape or rgb_ref.shape[:2] != depth_ref.shape:
        raise ValueError("reference and simulated frames must have the same shape")
    keep = np.ones(depth_ref.shape, bool) if robot_mask is None else ~np.asarray(robot_mask, bool)
    d_ok = keep & np.asarray(valid_depth_mask, bool) & np.isfinite(depth_ref) & (depth_ref > 0)
    sim_hit = depth_sim > 0
    d_use = d_ok & sim_hit
    n_px = depth_ref.size
    a, b = rgb_ref.astype(float) / 255.0, rgb_sim.astype(float) / 255.0
    n_rgb, n_depth = int(keep.sum()), int(d_use.sum())
    out = {
        "depth_l1_m": float(np.abs(depth_ref - depth_sim)[d_use].mean()) if n_depth else None,
        "n_depth": n_depth,
        "n_rgb": n_rgb,
        "depth_mask_coverage": round(n_depth / n_px, 4),
        "rgb_mask_coverage": round(n_rgb / n_px, 4),
        "depth_sim_no_hit": int((d_ok & ~sim_hit).sum()),
        "rgb_mae": None,
        "rgb_one_minus_ssim": None,
        "rgb_psnr_db": None,
    }
    if n_rgb:
        out["rgb_mae"] = float(np.abs(a - b)[keep].mean())
        mse = float(((a - b) ** 2)[keep].mean())
        out["rgb_psnr_db"] = None if mse == 0 else float(10 * math.log10(1.0 / mse))
        luma = np.array([0.299, 0.587, 0.114])
        out["rgb_one_minus_ssim"] = float(1.0 - _ssim_map(a @ luma, b @ luma)[keep].mean())
    return out


def side_by_side_png(path, rgb_ref, rgb_sim, depth_ref=None, depth_sim=None) -> None:
    """reference | twin (| depth reference | depth twin, grey, scaled to the reference's range) as one PNG."""
    from PIL import Image

    tiles = [np.asarray(rgb_ref), np.asarray(rgb_sim)]
    if depth_ref is not None and depth_sim is not None:
        hi = max(float(np.max(depth_ref)), 1e-6)
        for d in (depth_ref, depth_sim):
            g = (np.clip(np.asarray(d, float) / hi, 0, 1) * 255).astype(np.uint8)
            tiles.append(np.repeat(g[..., None], 3, axis=2))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def evaluate_frames(twin: dict, frames: list[dict], heldout_ids: list[str]) -> dict:
    """Score frames against the twin. Each frame is {id, rgb, depth, valid_depth_mask, K, cam_to_room[, robot_mask]}.
    Raises ValueError for any frame id not in heldout_ids, so a training frame cannot count. A twin whose
    blocks_claims holds 'realism' (a four-photo bundle) gets {"realism": None, "reason": ...} instead of numbers."""
    allowed = set(heldout_ids)
    bad = [str(f.get("id")) for f in frames if f.get("id") not in allowed]
    if bad:
        raise ValueError(f"frames not in the held-out manifest: {', '.join(bad)}")
    if "realism" in twin.get("blocks_claims", []):
        return {
            "realism": None,
            "reason": f"bundle {twin.get('room_id')!r} blocks the realism claim "
            f"(geometry_source {twin.get('geometry_source')!r}); no metrics reported",
        }
    per = []
    for f in frames:
        rgb = np.asarray(f["rgb"])
        rgb_sim, depth_sim = render_twin(twin, f["K"], f["cam_to_room"], rgb.shape[:2])
        s = score(rgb, f["depth"], rgb_sim, depth_sim, f["valid_depth_mask"], f.get("robot_mask"))
        per.append({"id": f["id"], **s})

    def pooled(key, weight):  # pixel-weighted mean over the frames that have the metric
        rows = [(p[key], p[weight]) for p in per if p[key] is not None]
        n = sum(w for _, w in rows)
        return sum(v * w for v, w in rows) / n if n else None

    return {
        "realism": {
            "frames": len(per),
            "heldout_only": True,
            "depth_l1_m": pooled("depth_l1_m", "n_depth"),
            "rgb_mae": pooled("rgb_mae", "n_rgb"),
            "rgb_one_minus_ssim": pooled("rgb_one_minus_ssim", "n_rgb"),
            "n_depth": sum(p["n_depth"] for p in per),
            "n_rgb": sum(p["n_rgb"] for p in per),
            "per_frame": per,
        }
    }


SHIFT_M, YAW_DEG = 0.15, 5.0  # the misaligned-pose control (perturb_pose)
MAX_SIDE = 192
MAX_FRAMES = 8
PNG_FRAMES = 3
MIN_CONFIDENCE = 2  # ARKit ARConfidenceLevel: 0 low, 1 medium, 2 high
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # ARKit camera axes (y up, z back) -> OpenCV (y down, z forward)
SCOPE = (
    "Schematic MuJoCo box render against reference frames: depth_l1_m measures geometry; rgb_mae, "
    "rgb_one_minus_ssim and rgb_psnr_db mostly measure missing texture and lighting, not layout."
)


class RealismError(ValueError):
    """The measurement cannot be made (or cannot be trusted); the message says why and what to do."""


def resolve_scan(scan: str | Path, scans_dir: str | Path = "scans") -> Path:
    """The scan folder: `scan` itself when it is a folder, else <scans_dir>/<scan>. Raises RealismError when neither
    exists."""
    p = Path(scan)
    if p.is_dir():
        return p
    q = Path(scans_dir) / str(scan)
    if q.is_dir():
        return q
    raise RealismError(f"no scan folder {scan!r} (looked at {p} and {q})")


def manifest_dir(scan_dir: Path) -> Path:
    """The folder holding the scan's frames_manifest.json: the scan folder or its frames/ subfolder. Raises RealismError
    when there is none, since a scan without posed frames cannot support a realism claim."""
    for d in (scan_dir, scan_dir / "frames"):
        if (d / rgbd.MANIFEST_NAME).is_file():
            return d
    raise RealismError(
        f"{scan_dir} has no {rgbd.MANIFEST_NAME}: realism needs posed RGB-D frames next to the scan. A RoomPlan-only "
        f"or four-photo scan cannot support a realism claim; record frames with the RGB-D recorder (docs/SCANNING.md)"
    )


def load_twin(scan_dir: Path, twin_path: str | Path | None = None) -> tuple[dict, str]:
    """(twin, where it came from). An explicit twin.json wins; else the scan's roomplan.json (a real capture),
    else its scan.json."""
    from arrival.scan import bundle as scan_bundle
    from arrival.scan import roomplan
    from arrival.twin.compile import compile_twin

    room = scan_dir.name
    if twin_path is not None:
        p = Path(twin_path)
        if not p.is_file():
            raise RealismError(f"--twin {p} does not exist")
        try:
            twin = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RealismError(f"--twin {p} is not readable JSON: {e}") from e
        missing = [
            k
            for k in ("walls", "fixtures", "objects")
            if not isinstance(twin, dict) or not isinstance(twin.get(k), list)
        ]
        if missing:
            raise RealismError(
                f"--twin {p} is not a compiled twin (no {', '.join(missing)} list); pass out/twins/<room_id>/twin.json"
            )
        return twin, str(p)
    for name, load in (("roomplan.json", lambda p: roomplan.load(p, room)), ("scan.json", scan_bundle.load)):
        p = scan_dir / name
        if p.is_file():
            try:
                return compile_twin(load(p)), str(p)
            except ValueError as e:
                raise RealismError(f"{p} cannot be compiled into a twin: {e}") from e
    raise RealismError(f"{scan_dir} has neither roomplan.json nor scan.json (or pass --twin)")


def _nearest(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample an HxW array to (width, height) = size by nearest neighbour: no interpolated depth."""
    h, w = a.shape[:2]
    tw, th = size
    ys = np.minimum((np.arange(th) * h / th + h / (2 * th)).astype(int), h - 1)
    xs = np.minimum((np.arange(tw) * w / tw + w / (2 * tw)).astype(int), w - 1)
    return a[np.ix_(ys, xs)]


def _open_image(path: Path, mode: str, frame_id: str):
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.convert(mode)
    except OSError as e:
        raise RealismError(f"frame {frame_id}: {path} is not a readable image: {e}") from e


def load_frame(m: rgbd.Manifest, f: rgbd.Frame, max_side: int = MAX_SIDE, min_confidence: int = MIN_CONFIDENCE) -> dict:
    """One manifest frame as the dict evaluate_frames takes: RGB, depth in metres, valid-depth mask and K at the
    working size, and cam_to_room in OpenCV axes (the convention of eval/realism.py)."""
    from PIL import Image

    rgb = _open_image(f.rgb, "RGB", f.id)
    if rgb.size != tuple(m.rgb_size):
        raise RealismError(
            f"frame {f.id}: rgb is {rgb.size[0]}x{rgb.size[1]}, the manifest says {m.rgb_size[0]}x{m.rgb_size[1]}"
        )
    try:
        depth = rgbd.load_depth(m, f)
    except (OSError, EOFError, ValueError) as e:
        raise RealismError(f"frame {f.id}: depth {f.depth} cannot be used: {e}") from e
    valid = np.isfinite(depth) & (depth > 0)
    if f.confidence is not None:
        conf = np.asarray(_open_image(f.confidence, "L", f.id))
        if conf.shape != depth.shape:
            raise RealismError(f"frame {f.id}: confidence map is {conf.shape[::-1]}, depth is {depth.shape[::-1]}")
        if np.any(conf > 2):
            raise RealismError(f"frame {f.id}: confidence map must use ARKit levels 0, 1 or 2")
        valid &= conf >= min_confidence
    w, h = m.rgb_size
    scale = min(1.0, max_side / max(w, h))
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    if (tw, th) != (w, h):
        rgb = rgb.resize((tw, th), Image.LANCZOS)
    K = f.K.copy()
    K[0] *= tw / w
    K[1] *= th / h
    depth_t = _nearest(np.where(valid, depth, 0.0), (tw, th)).astype(np.float32)
    return {
        "id": f.id,
        "rgb": np.asarray(rgb),
        "depth": depth_t,
        "valid_depth_mask": depth_t > 0,
        "K": K,
        "cam_to_room": f.camera_to_room @ ARKIT_TO_CV,
    }


def _pick(frames: list, n: int) -> list:
    if len(frames) <= n:
        return frames
    return [frames[round(i * (len(frames) - 1) / (n - 1))] for i in range(n)] if n > 1 else frames[:1]


def _stack_png(path: Path, rows: list[Path]) -> None:
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in rows]
    out = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)))
    y = 0
    for im in ims:
        out.paste(im, (0, y))
        y += im.height
    out.save(path)


def run_realism(
    scan: str | Path,
    out_dir: str | Path = "out",
    scans_dir: str | Path = "scans",
    twin_path: str | Path | None = None,
    max_side: int = MAX_SIDE,
    max_frames: int = MAX_FRAMES,
    png_frames: int = PNG_FRAMES,
    min_confidence: int = MIN_CONFIDENCE,
    log=print,
    camera_transform=None,
) -> dict:
    """Score the held-out frames of a scan against the twin and write out/eval/<room>_realism.json and .png.
    Raises RealismError when no honest measurement can be made. Returns the report dict (with `path`, `png`)."""
    if type(max_side) is not int or max_side < 16 or type(max_frames) is not int or max_frames < 1:
        raise RealismError("--max-side must be at least 16 and --max-frames at least 1")
    if type(png_frames) is not int or png_frames < 0:
        raise RealismError("--png-frames must be a nonnegative integer")
    if type(min_confidence) is not int or min_confidence not in (0, 1, 2):
        raise RealismError("--min-confidence must be an ARKit confidence level: 0, 1, or 2")
    transform = None
    if camera_transform is not None:
        try:
            transform = np.asarray(camera_transform, dtype=float)
        except (TypeError, ValueError) as e:
            raise RealismError("camera_transform must be a finite 4x4 transform") from e
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise RealismError("camera_transform must be a finite 4x4 homogeneous transform")
    scan_dir = resolve_scan(scan, scans_dir)
    mdir = manifest_dir(scan_dir)
    try:
        manifest = rgbd.load_manifest(mdir)
    except rgbd.ManifestError as e:
        raise RealismError(f"{mdir / rgbd.MANIFEST_NAME} is not usable: {e}") from e
    twin, twin_source = load_twin(scan_dir, twin_path)
    heldout = [f for f in manifest.frames if f.split == "heldout"]
    train = [f for f in manifest.frames if f.split != "heldout"]
    if not heldout:
        raise RealismError(
            f"{mdir / rgbd.MANIFEST_NAME} has no held-out frame ({len(train)} train frames only); "
            f"training frames cannot be used as evidence"
        )
    used = _pick(heldout, max_frames)
    frames = [load_frame(manifest, f, max_side, min_confidence) for f in used]
    if transform is not None:
        for frame in frames:
            frame["cam_to_room"] = transform @ frame["cam_to_room"]
    ids = [f.id for f in used]
    size = frames[0]["rgb"].shape[:2]
    try:
        real = evaluate_frames(twin, frames, [f.id for f in heldout])
        control_frames = [{**f, "cam_to_room": perturb_pose(f["cam_to_room"], SHIFT_M, YAW_DEG)} for f in frames]
        control = evaluate_frames(twin, control_frames, [f.id for f in heldout])
    except ValueError as e:
        raise RealismError(str(e)) from e

    out_dir = Path(out_dir)
    stem = f"{twin.get('room_id') or scan_dir.name}_realism"
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "room_id": twin.get("room_id") or scan_dir.name,
        "scan_dir": str(scan_dir),
        "twin_source": twin_source,
        "geometry_source": twin.get("geometry_source"),
        "frames": {
            "manifest": str(mdir / rgbd.MANIFEST_NAME),
            "manifest_sha256": sha256_file(mdir / rgbd.MANIFEST_NAME),
            "total": len(manifest.frames),
            "heldout": len(heldout),
            "scored": ids,
            "train_excluded": len(train),
            "working_size_hw": list(size),
        },
        "masks": {
            "invalid_depth": f"reference depth not finite, 0, or confidence < {min_confidence}",
            "robot": "hidden in the twin render; the recorded frames contain no robot, so no robot mask",
        },
        "scope": SCOPE,
        "realism": real["realism"],
        "control": {"kind": "misaligned_pose", "shift_m": SHIFT_M, "yaw_deg": YAW_DEG, "realism": control["realism"]},
        "code": git_state(),
    }
    if real["realism"] is None:  # a four-photo bundle: the claim is blocked, no numbers
        report["reason"] = real["reason"]
        report["control"] = None
        report["alignment_check"] = None
        report["valid"] = False
    else:
        r, c = real["realism"], control["realism"]
        rises = {k: (r[k] is not None and c[k] is not None and c[k] > r[k]) for k in ("depth_l1_m", "rgb_mae")}
        report["alignment_check"] = {"error_rises_under_misalignment": rises, "passed": bool(rises["depth_l1_m"])}
        report["valid"] = bool(rises["depth_l1_m"])
        if not report["valid"]:
            report["reason"] = (
                f"depth error did not rise when the camera was moved {SHIFT_M} m and turned {YAW_DEG} "
                f"degrees (real {r['depth_l1_m']}, control {c['depth_l1_m']}): the poses or the twin's "
                f"frame are off, so these numbers must not be quoted"
            )
    path = eval_dir / f"{stem}.json"
    report["path"] = str(path)
    png = None
    if real["realism"] is not None:
        rows = []
        pose_by_id = {f["id"]: f for f in frames}
        for index, fid in enumerate(ids[:png_frames]):
            f = pose_by_id[fid]
            rgb_sim, depth_sim = render_twin(twin, f["K"], f["cam_to_room"], f["rgb"].shape[:2])
            row = eval_dir / f"{stem}_{index:03d}.png"
            side_by_side_png(row, f["rgb"], rgb_sim, f["depth"] * f["valid_depth_mask"], depth_sim)
            rows.append(row)
        if rows:
            png = eval_dir / f"{stem}.png"
            _stack_png(png, rows)
    report["png"] = str(png) if png else None
    path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    log(f"[realism] {report['room_id']}: {len(ids)}/{len(heldout)} held-out frames, {len(train)} train excluded")
    return report


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def summary(report: dict) -> str:
    """The realism report in two or three lines: the held-out scores, the misaligned-pose control's scores, and NOT
    VALID with the reason when the control failed."""
    r = report["realism"]
    if r is None:
        return f"realism: null ({report['reason']})"

    def f(v, nd=4):
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"

    c = report["control"]["realism"]
    lines = [
        f"realism {report['room_id']}: {r['frames']} held-out frame(s), depth L1 {f(r['depth_l1_m'])} m, "
        f"RGB MAE {f(r['rgb_mae'])}, 1-SSIM {f(r['rgb_one_minus_ssim'])}",
        f"control (camera moved {SHIFT_M} m, turned {YAW_DEG} deg): depth L1 {f(c['depth_l1_m'])} m, "
        f"RGB MAE {f(c['rgb_mae'])}",
    ]
    if not report["valid"]:
        lines.append(f"NOT VALID: {report['reason']}")
    return "\n".join(lines)


METRICS = ("depth_l1_m", "rgb_mae", "rgb_one_minus_ssim", "n_depth", "n_rgb", "frames")


def measure(job_dir: Path, bundle: dict, twin: dict, log=print) -> dict:
    """The worker's optional realism stage: score the job's uploaded held-out frames against its compiled twin and
    return the status block the job records. Never raises: a failure is recorded as state "error"."""
    status = {
        "state": "not_measured",
        "metrics": None,
        "consumed": False,
        "preview": False,
        "reason": "No RGB-D frames uploaded.",
    }
    frames_dir = job_dir / "frames"
    if not frames_dir.is_dir():
        return status
    output = job_dir / "realism"
    try:
        output.mkdir(exist_ok=True)
        manifest = rgbd.load_manifest(frames_dir)
        source_id = bundle.get("source_room_id")
        if source_id and source_id != manifest.room_id:
            status["reason"] = "The scan and frame bundle name different rooms."
            return status
        if not any(f.split == "heldout" for f in manifest.frames):
            status["reason"] = "No held-out frames were supplied."
            return status
        if "realism" in twin.get("blocks_claims", []):
            status["reason"] = "This scan does not support a realism measurement."
            return status
        transform = np.eye(4)
        if "arkit_world_to_room" in bundle:
            transform = np.asarray(bundle["arkit_world_to_room"]) @ np.linalg.inv(manifest.arkit_world_to_room)
        alignment = {
            "source_room_id": source_id,
            "manifest_room_id": manifest.room_id,
            "association": "matching_room_id" if source_id else "uploaded_together",
            "manifest_room_to_twin": transform.tolist(),
        }
        (output / "alignment.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
        twin_path = output / "twin.json"
        twin_path.write_text(json.dumps(twin, indent=2), encoding="utf-8")
        report = run_realism(job_dir, out_dir=output, twin_path=twin_path, camera_transform=transform, log=log)
        status.update(
            consumed=True,
            state="measured" if report["valid"] else "invalid",
            reason=None if report["valid"] else "The camera-offset control did not increase depth error.",
            preview=bool(report["png"]),
            frame_count=len(report["frames"]["scored"]),
            association=alignment["association"],
        )
        if report["valid"]:
            status["metrics"] = {key: report["realism"][key] for key in METRICS}
        if report["png"]:
            # The web route serves only this fixed artifact, never a client-supplied path.
            (output / "comparison.png").write_bytes(Path(report["png"]).read_bytes())
    except Exception:
        status.update(state="error", metrics=None, preview=False, reason="Frame comparison failed.")
        with contextlib.suppress(OSError):
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"[work] {job_dir.name}: optional realism measurement failed")
    return status
