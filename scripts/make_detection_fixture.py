"""Render the C4 detection fixtures: posed RGB-D frames of hero_kitchen's twin with and without one mug (TWN-5).

    python scripts/make_detection_fixture.py            # writes tests/fixtures/detection/{mug,empty}/

The twin is scans/hero_kitchen compiled with its objects removed; `mug` adds one catalog mug at MUG_POSITION.
Each frame is rendered with `arrival.eval.realism.render_twin` (MuJoCo, robot and people hidden) and stored as a
CAP-2a bundle: RGB PNG, depth as 16-bit millimetre PNG, ARKit camera axes. Synthetic renders only, never a real
capture. For each frame the script prints the detection cache key and the mug's pixel box (its bounding box
projected into the frame, and the rendered pixels that differ from the empty room), which is what the hand-written
replies in tests/fixtures/detection/replies.json were written from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrival.eval.realism import look_at, render_twin  # noqa: E402
from arrival.scan import rgbd  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402
from arrival.twin.detect import DetectionClient  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "detection"
MUG_POSITION = [3.2, 0.45, 0.97]  # on counter_b (top at 0.92 m), clear of the stove
MUG_SIZE = [0.08, 0.08, 0.10]
W, H, F = 128, 96, 110.0
TARGET = (3.2, 0.45, 0.95)
EYES = [(3.2, 1.15, 1.45), (2.7, 1.05, 1.40), (3.75, 1.10, 1.40)]
_CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])


def fixture_twin(with_mug: bool) -> dict:
    twin = compile_twin(load_room("hero_kitchen", ROOT / "scans"))
    twin["objects"] = []
    if with_mug:
        twin["objects"].append(
            {"id": "mug_1", "catalog_id": "mug", "label": "mug", "size": MUG_SIZE, "position": MUG_POSITION, "yaw": 0.0}
        )
    return twin


def write(name: str, twin: dict) -> list[dict]:
    root = OUT / name
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    frames, renders = [], []
    for i, eye in enumerate(EYES):
        pose = look_at(eye, TARGET)
        rgb, depth = render_twin(twin, K, pose, (H, W))
        fid = f"f{i:03d}"
        Image.fromarray(rgb).save(root / "rgb" / f"{fid}.png")
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(root / "depth" / f"{fid}.png")
        frames.append(
            {
                "id": fid,
                "timestamp": 1.0 + i,
                "rgb": f"rgb/{fid}.png",
                "depth": f"depth/{fid}.png",
                "intrinsics": {"fx": F, "fy": F, "cx": W / 2, "cy": H / 2},
                "camera_to_room": (pose @ _CV_TO_ARKIT).ravel().tolist(),
                "split": "train",
            }
        )
        renders.append(depth)
    manifest = {
        "schema_version": rgbd.SCHEMA_VERSION,
        "room_id": "hero_kitchen",
        "axes": {"camera": rgbd.CAMERA_AXES, "room": rgbd.ROOM_AXES},
        "arkit_world_to_room": np.eye(4).ravel().tolist(),
        "rgb_size": [W, H],
        "depth_size": [W, H],
        "depth_unit": "mm",
        "frames": frames,
    }
    (root / rgbd.MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return [{"frame": f, "depth": d} for f, d in zip(frames, renders)]


def main() -> int:
    empty = write("empty", fixture_twin(False))
    mug = write("mug", fixture_twin(True))
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    corners = np.array(
        [
            [
                MUG_POSITION[0] + sx * MUG_SIZE[0] / 2,
                MUG_POSITION[1] + sy * MUG_SIZE[1] / 2,
                MUG_POSITION[2] + sz * MUG_SIZE[2] / 2,
            ]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    for name, rows in (("empty", empty), ("mug", mug)):
        for r, e in zip(rows, empty):
            f = r["frame"]
            key = DetectionClient.key(OUT / name / f["rgb"])
            T = np.asarray(f["camera_to_room"]).reshape(4, 4)
            uv, _ = rgbd.project(corners, K, T)
            box = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
            diff = np.argwhere(np.abs(np.round(r["depth"] * 1000) - np.round(e["depth"] * 1000)) > 0)
            seen = (
                None
                if not len(diff)
                else [
                    int(diff[:, 1].min()),
                    int(diff[:, 0].min()),
                    int(diff[:, 1].max()) + 1,
                    int(diff[:, 0].max()) + 1,
                ]
            )
            print(
                f"{name}/{f['id']} key={key} projected_box={[round(b, 1) for b in box]} rendered_px={seen} "
                f"({len(diff)} px)"
            )
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
