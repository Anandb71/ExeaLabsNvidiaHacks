import http.client
import io
import json
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import SCANS
from PIL import Image

from arrival.eval.realism import measure
from arrival.jobs.queue import read_job, write_status
from arrival.jobs.worker import load_upload, run_once
from arrival.scan import rgbd, roomplan
from arrival.web.app import make_server


def capture(root, twin, room_id, room_transform=None, frame_id="heldout"):
    from arrival.eval.realism import look_at, render_twin

    root.mkdir(parents=True)
    K = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    pose = look_at((3.6, 2.5, 1.5), (2.0, 1.0, 0.8))
    rgb, depth = render_twin(twin, K, pose, (48, 64))
    confidence = np.full(depth.shape, 2, dtype=np.uint8)
    depth[:, :8] += 5
    confidence[:, :8] = 0
    Image.fromarray(rgb).save(root / "reference.png")
    np.save(root / "depth.npy", depth)
    Image.fromarray(confidence).save(root / "confidence.png")
    camera = pose @ np.diag([1.0, -1.0, -1.0, 1.0])
    if room_transform is not None:
        camera = room_transform @ camera
    frames = [
        {
            "id": name,
            "timestamp": float(i),
            "rgb": "reference.png",
            "depth": "depth.npy",
            "confidence": "confidence.png",
            "intrinsics": {"fx": 60, "fy": 60, "cx": 32, "cy": 24},
            "camera_to_room": camera.ravel().tolist(),
            "split": split,
        }
        for i, (name, split) in enumerate((("train", "train"), (frame_id, "heldout")))
    ]
    manifest = {
        "schema_version": 1,
        "room_id": room_id,
        "axes": {"camera": rgbd.CAMERA_AXES, "room": rgbd.ROOM_AXES},
        "arkit_world_to_room": np.eye(4).ravel().tolist(),
        "rgb_size": [64, 48],
        "depth_size": [64, 48],
        "depth_unit": "m",
        "frames": frames,
    }
    (root / rgbd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def server(tmp_path):
    out = tmp_path / "out"
    srv = make_server(port=0, out_dir=out, scans_dir=SCANS, public=True, rate_limit=0)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield srv, out
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()


def request(srv, path, method="GET", body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
    conn.request(method, path, body, headers or {})
    response = conn.getresponse()
    payload, response_headers, status = response.read(), dict(response.getheaders()), response.status
    conn.close()
    return status, payload, response_headers


def upload(srv, scan, frames):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in frames.iterdir():
            bundle.write(path, path.name)
    boundary = "arrival-realism-test"
    body = b""
    for field, name, contents in (("scan", "kitchen.json", scan), ("frames", "frames.zip", archive.getvalue())):
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n\r\n').encode()
        body += contents + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    code, payload, headers = request(
        srv, "/jobs", "POST", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    assert code == 202, payload
    return json.loads(payload), {"Cookie": headers["Set-Cookie"].split(";")[0]}


def test_upload_worker_comparison_and_private_preview(server, tmp_path, hero_twin):
    srv, out = server
    frames = tmp_path / "capture"
    capture(frames, hero_twin, "hero_kitchen", frame_id="../outside")
    job, owner = upload(srv, (SCANS / "hero_kitchen" / "scan.json").read_bytes(), frames)
    finished = run_once(out, episodes=1, families=["open_drawer"], people=0, res=32, log=lambda *_: None)
    assert finished["status"] == "done", finished
    comparison = finished["realism"]
    assert comparison["state"] == "measured", comparison
    assert comparison["metrics"]["depth_l1_m"] < 1e-5
    assert comparison["metrics"]["frames"] == 1
    assert finished["frames"]["consumed"] and finished["frames"]["used_for"] == "realism"
    base = f"/jobs/{job['id']}/realism"
    code, page, _ = request(srv, base, headers=owner)
    assert code == 200 and b"Depth error (geometry)" in page and b"/realism/preview" in page
    code, report, _ = request(srv, base + "/report", headers=owner)
    assert code == 200 and json.loads(report)["metrics"] == comparison["metrics"]
    assert str(out).encode() not in report and b"manifest_sha256" not in report
    code, image, headers = request(srv, base + "/preview", headers=owner)
    assert code == 200 and image.startswith(b"\x89PNG") and headers["Cache-Control"] == "no-store"
    assert Image.open(io.BytesIO(image)).width == 64 * 4
    for suffix in ("", "/report", "/preview"):
        assert request(srv, base + suffix)[0] == 404
        assert request(srv, base + suffix, headers={"Cookie": "arrival_owner=" + "a" * 64})[0] == 404
    assert request(srv, base + "/../../upload.bin", headers=owner)[0] == 404
    assert request(srv, base + "/preview/extra", headers=owner)[0] == 404
    job_dir = out / "jobs" / job["id"]
    assert not (job_dir / "outside.png").exists()
    (report_path,) = (job_dir / "realism" / "eval").glob("*.json")
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    assert raw["frames"]["scored"] == ["../outside"] and raw["frames"]["train_excluded"] == 1

    # A failed retry must not expose the previous successful comparison.
    retry = read_job(out / "jobs", job["id"])
    retry["status"] = "queued"
    write_status(job_dir, retry)
    (job_dir / "upload.bin").write_bytes(b"invalid JSON")
    failed = run_once(out, log=lambda *_: None)
    assert failed["status"] == "failed" and failed["realism"] is None
    assert not failed["frames"]["consumed"]
    assert request(srv, base + "/preview", headers=owner)[0] == 404


@pytest.mark.parametrize("case", ["absent", "room_mismatch", "train_only", "corrupt", "blocked"])
def test_unavailable_measurements_are_explicit(tmp_path, hero_bundle, hero_twin, case):
    job = tmp_path / "job"
    job.mkdir()
    bundle = {**hero_bundle, "source_room_id": "synthetic_room"}
    if case != "absent":
        rgbd.write_synthetic_bundle(job / "frames")
    if case == "room_mismatch":
        bundle["source_room_id"] = "a_different_room"
    elif case == "train_only":
        path = job / "frames" / rgbd.MANIFEST_NAME
        manifest = json.loads(path.read_text())
        for frame in manifest["frames"]:
            frame["split"] = "train"
        path.write_text(json.dumps(manifest))
    elif case == "corrupt":
        (job / "frames" / rgbd.MANIFEST_NAME).write_text("{")
    elif case == "blocked":
        hero_twin["blocks_claims"] = ["realism"]
    comparison = measure(job, bundle, hero_twin)
    assert comparison["state"] in ("not_measured", "error") and comparison["metrics"] is None
    assert not comparison["consumed"] and not comparison["preview"]
    assert comparison["reason"] and str(job) not in comparison["reason"]


def test_control_failure_cannot_be_published_as_a_measurement(tmp_path, hero_bundle, hero_twin):
    job = tmp_path / "job"
    capture(job / "frames", hero_twin, "hero_kitchen")
    np.save(job / "frames" / "depth.npy", np.zeros((48, 64), dtype=np.float32))
    comparison = measure(job, {**hero_bundle, "source_room_id": "hero_kitchen"}, hero_twin)
    assert comparison["state"] == "invalid" and comparison["metrics"] is None
    assert comparison["consumed"]


def test_roomplan_camera_transform_tracks_normalization(tmp_path):
    from arrival.twin.compile import compile_twin

    raw = json.loads((SCANS / "hero_kitchen" / "roomplan.json").read_text())
    original = roomplan.to_bundle(raw, "room")
    angle = 0.4
    c, s = np.cos(angle), np.sin(angle)
    motion = np.array([[c, 0, s, 1.2], [0, 1, 0, 0.7], [-s, 0, c, -2.0], [0, 0, 0, 1]])
    for key in (*roomplan.SURFACES, "objects"):
        for item in raw.get(key, []):
            matrix = np.asarray(item["transform"]).reshape(4, 4, order="F")
            item["transform"] = (motion @ matrix).ravel(order="F").tolist()
    job = tmp_path / "job"
    job.mkdir()
    (job / "upload.bin").write_text(json.dumps(raw))
    normalized = load_upload(job / "upload.bin", "renamed_job")
    expected = np.asarray(original["arkit_world_to_room"])
    actual = np.asarray(normalized["arkit_world_to_room"])
    assert np.allclose(actual @ motion, expected)
    twin = compile_twin(normalized)
    # The manifest uses another valid Z-up frame instead of the compiler's origin.
    frame_basis = np.eye(4)
    frame_basis[:3, 3] = [2.0, -1.0, 0.5]
    frame_map = frame_basis @ np.linalg.inv(actual)
    manifest = capture(job / "frames", twin, "capture-room", room_transform=frame_map)
    manifest["arkit_world_to_room"] = frame_basis.ravel().tolist()
    (job / "frames" / rgbd.MANIFEST_NAME).write_text(json.dumps(manifest))
    comparison = measure(job, normalized, twin, log=lambda *_: None)
    assert comparison["state"] == "measured", comparison
    assert comparison["metrics"]["depth_l1_m"] < 1e-5
    assert comparison["association"] == "uploaded_together"


def test_worker_compares_before_adding_synthetic_objects(tmp_path, monkeypatch):
    from arrival.jobs.queue import create_job

    raw = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text())
    raw["objects"] = []
    out = tmp_path / "out"
    job = create_job(out / "jobs", "room.json", data=json.dumps(raw).encode())
    seen = []

    def compare(job_dir, bundle, twin, log):
        seen.append(json.loads(json.dumps(twin)))
        return {"state": "not_measured", "metrics": None, "consumed": False, "preview": False}

    monkeypatch.setattr("arrival.eval.realism.measure", compare)
    finished = run_once(out, episodes=1, families=["open_drawer"], people=0, res=32, log=lambda *_: None)
    assert finished["status"] == "done", finished
    assert seen[0]["objects"] == []
    saved = json.loads((Path(finished["twin_dir"]) / "twin.json").read_text())
    assert saved["objects"]
    assert finished["id"] == job["id"]


@pytest.mark.parametrize("failure", ["renderer", "directory", "diagnostic"])
def test_comparison_failure_does_not_discard_dataset(tmp_path, monkeypatch, hero_twin, failure):
    from arrival.export.validate import validate
    from arrival.jobs.queue import create_job

    out = tmp_path / "out"
    job = create_job(out / "jobs", "room.json", data=(SCANS / "hero_kitchen" / "scan.json").read_bytes())
    capture(out / "jobs" / job["id"] / "frames", hero_twin, "hero_kitchen")

    def fail(*args, **kwargs):
        raise RuntimeError("broken renderer")

    monkeypatch.setattr("arrival.eval.realism.run_realism", fail)
    output = out / "jobs" / job["id"] / "realism"
    if failure == "directory":
        mkdir = Path.mkdir

        def deny_directory(path, *args, **kwargs):
            if path == output:
                raise PermissionError("cannot create comparison directory")
            return mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", deny_directory)
    elif failure == "diagnostic":
        write_text = Path.write_text

        def deny_diagnostic(path, *args, **kwargs):
            if path == output / "error.txt":
                raise PermissionError("cannot write comparison diagnostic")
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", deny_diagnostic)
    finished = run_once(out, episodes=1, families=["open_drawer"], people=0, res=32, log=lambda *_: None)
    assert finished["status"] == "done", finished
    assert finished["realism"]["state"] == "error" and finished["realism"]["metrics"] is None
    assert not validate(finished["dataset_dir"])
