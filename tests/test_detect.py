"""C4 (TWN-5): objects detected in posed RGB-D frames and placed in the twin, offline.

The frames under tests/fixtures/detection are MuJoCo renders of hero_kitchen's twin with and without one mug at
MUG (scripts/make_detection_fixture.py). replies.json is a hand-written stand-in for the model's replies, not a
recorded one: no Token Factory call has been made for this feature."""

import json
import math
import shutil
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO, SCANS
from PIL import Image

from arrival.scan import rgbd
from arrival.twin import detect

FIXTURES = REPO / "tests" / "fixtures" / "detection"
REPLIES = FIXTURES / "replies.json"
MUG = [3.2, 0.45, 0.97]  # the mug's centre in the rendered twin
XY_TOL, Z_TOL = 0.03, 0.01  # under the mug's 4 cm radius; z sits on the counter top


@pytest.fixture
def room(hero_bundle):
    from arrival.twin.compile import compile_twin

    twin = compile_twin(hero_bundle)
    twin["objects"] = []
    return hero_bundle, twin


def run(name, room, client=None):
    bundle, twin = room
    return detect.detect_objects(
        FIXTURES / name, twin, bundle, client=client or detect.DetectionClient(REPLIES), log=lambda *_: None
    )


class FakeClient:
    """Replies by frame id, as a model would give them; counts reads."""

    def __init__(self, replies):
        self.replies, self.model, self.reads = replies, "fake", 0

    def read(self, frame):
        self.reads += 1
        return detect.parse_reply(self.replies.get(frame.id, {"objects": []})), "fake"


def test_fixture_replies_are_labelled_hand_written():
    doc = json.loads(REPLIES.read_text(encoding="utf-8"))
    assert "Hand-written" in doc["about"] and "no Token Factory call" in doc["about"]
    assert {e["source"] for e in doc["entries"].values()} == {"hand_written"}
    for name in ("mug", "empty"):  # every fixture frame has a reply, keyed by its image
        for f in rgbd.load_manifest(FIXTURES / name).frames:
            assert doc["entries"][detect.DetectionClient.key(f.rgb)]["frame"] == f"{name}/{f.id}"


def test_known_mug_becomes_one_correctly_placed_object(room):
    res = run("mug", room)
    assert res["state"] == "detected" and res["frames_read"] == 3 and res["reply_sources"] == {"hand_written": 3}
    assert res["rejected"] == []
    (obj,) = res["objects"]  # three views of one mug merge into one object
    assert obj["label"] == obj["catalog_id"] == "mug" and obj["provenance"] == "detected"
    assert obj["size"] == [0.08, 0.08, 0.10]
    assert math.dist(obj["position"][:2], MUG[:2]) < XY_TOL
    assert abs(obj["position"][2] - MUG[2]) < Z_TOL
    assert obj["detection"]["frames"] == ["f000", "f001", "f002"] and len(obj["detection"]["evidence"]) == 3


@pytest.mark.parametrize("pad", [3, 8])
def test_loose_boxes_place_the_same_mug(room, pad):
    # a model's box is looser than the rendered extent and takes in the counter, the backsplash and the wall
    doc = json.loads(REPLIES.read_text(encoding="utf-8"))["entries"]
    boxes = {e["frame"].split("/")[1]: e["reply"]["objects"][0]["box"] for e in doc.values() if e["reply"]["objects"]}
    fake = FakeClient(
        {
            f: {
                "objects": [
                    {"label": "mug", "box": [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], "confidence": 0.9}
                ]
            }
            for f, b in boxes.items()
        }
    )
    res = run("mug", room, client=fake)
    (obj,) = res["objects"]
    assert math.dist(obj["position"][:2], MUG[:2]) < XY_TOL and abs(obj["position"][2] - MUG[2]) < Z_TOL
    assert obj["detection"]["frames"] == ["f000", "f001", "f002"] and res["rejected"] == []


def _boxes_of(name, lo, hi):
    """Each frame's pixel box around the room-frame block lo..hi, from the frame's own pose."""
    m = rgbd.load_manifest(FIXTURES / name)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    out = {}
    for f in m.frames:
        uv, _ = rgbd.project(corners, f.K, f.camera_to_room)
        out[f.id] = [
            math.floor(uv[:, 0].min()),
            math.floor(uv[:, 1].min()),
            math.ceil(uv[:, 0].max()),
            math.ceil(uv[:, 1].max()),
        ]
    return out


@pytest.mark.parametrize("label", ["mug", "cup", "bottle", "book"])
def test_empty_room_places_nothing_for_loose_backsplash_or_wall_boxes(room, label):
    # nothing stands on the counter, so no box, however loose or wherever it is drawn, may become an object
    regions = {
        "backsplash": ((3.16, 0.0, 0.93), (3.24, 0.10, 1.03)),
        "wall": ((3.10, -0.02, 1.0), (3.30, 0.08, 1.3)),
        "bare_front": ((3.1, 0.5, 0.92), (3.3, 0.7, 0.95)),
        "mug_spot": ((3.16, 0.41, 0.92), (3.24, 0.49, 1.02)),
    }
    for name, (lo, hi) in regions.items():
        for pad in (0, 8, 20):
            boxes = _boxes_of("empty", lo, hi)
            fake = FakeClient(
                {
                    f: {
                        "objects": [
                            {"label": label, "box": [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], "confidence": 0.9}
                        ]
                    }
                    for f, b in boxes.items()
                }
            )
            res = run("empty", room, client=fake)
            assert res["objects"] == [], (name, pad, res["objects"])
            assert len(res["rejected"]) == 3


def test_two_boxes_on_one_mug_in_one_frame_are_one_object(room):
    tight = {"label": "mug", "box": [57, 38, 69, 54], "confidence": 0.9}
    loose = {"label": "mug", "box": [49, 30, 77, 62], "confidence": 0.8}
    res = run("mug", room, client=FakeClient({"f000": {"objects": [loose, tight]}}))
    (obj,) = res["objects"]
    assert obj["detection"]["evidence"][0]["box"] == tight["box"]
    assert [(r["box"], r["reason"]) for r in res["rejected"]] == [(loose["box"], "same_object")]


def _hit(frame, x, y, label="mug", confidence=0.9):
    return {"frame": frame, "label": label, "box": [0, 0, 1, 1], "confidence": confidence, "position": [x, y, 0.97]}


def test_merge_keeps_objects_one_frame_shows_apart(room):
    # two mugs 6 cm apart seen together in two frames stay two objects; a frame never shows one object twice
    hits = [_hit("f000", 3.2, 0.42), _hit("f000", 3.2, 0.48), _hit("f001", 3.2, 0.43), _hit("f001", 3.2, 0.49)]
    out = detect.merge(hits, room[1])
    assert sorted(round(o["position"][1], 3) for o in out) == [0.425, 0.485]
    assert all(o["detection"]["frames"] == ["f000", "f001"] for o in out)


def test_merged_position_must_clear_the_keep_outs(room):
    # each view of the cup is clear of the sink, one beside it and one in front of it, but their mean falls on the
    # sink's corner: the space around a sink is not convex, so the merged position is checked again
    hits = [_hit("f000", 1.265, 0.62, "cup"), _hit("f001", 1.31, 0.69, "cup")]
    support = detect.support_under((1.29, 0.65), room[1])
    assert [detect.clearance(h["position"][:2], "cup", support, room[1]) for h in hits] == [None, None]
    assert detect.clearance((1.2875, 0.655), "cup", support, room[1]) == "on_fixture"
    rejected = []
    assert detect.merge(hits, room[1], rejected) == []
    assert [(r["frame"], r["reason"]) for r in rejected] == [("f000", "on_fixture"), ("f001", "on_fixture")]


def test_merge_joins_views_of_one_object_and_picks_the_nearest_cluster(room):
    # one mug seen 8 cm apart in two frames is one object, as DEDUP_M intends
    (one,) = detect.merge([_hit("f000", 3.20, 0.45), _hit("f001", 3.28, 0.45)], room[1])
    assert one["detection"]["frames"] == ["f000", "f001"]
    # a third view 7 cm from one cluster and 9 cm from the other joins the nearer one
    hits = [_hit("f000", 3.10, 0.45), _hit("f000", 3.26, 0.45), _hit("f001", 3.19, 0.45)]
    out = sorted(detect.merge(hits, room[1]), key=lambda o: o["position"][0])
    assert [o["detection"]["frames"] for o in out] == [["f000"], ["f000", "f001"]]


def test_room_without_the_mug_gives_no_object(room):
    res = run("empty", room)
    assert res["state"] == "none_found" and res["frames_read"] == 3 and res["objects"] == []


def test_fill_objects_uses_detections_else_labelled_defaults(room):
    from arrival.twin.compile import fill_objects

    bundle, twin = room
    found = json.loads(json.dumps(twin))
    fill_objects(found, run("mug", room), seed=0)
    assert [o["provenance"] for o in found["objects"]] == ["detected"]
    assert found["provenance"]["generators"]["objects"] == {"detected": 1, "authored": 0, "default_synthetic": 0}
    assert found["object_detection"]["state"] == "detected" and "objects" not in found["object_detection"]

    fill_objects(twin, run("empty", room), seed=0)
    assert twin["objects"] and {o["provenance"] for o in twin["objects"]} == {"default_synthetic"}
    assert twin["object_detection"]["state"] == "none_found"


def test_no_credentials_reads_nothing_and_invents_nothing(room, monkeypatch):
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("no request may be made"))
    res = run("mug", room, client=detect.DetectionClient(None))
    assert res["state"] == "not_run" and res["objects"] == [] and res["frames_read"] == 0
    assert res["reply_sources"] == {"not_configured": 3} and "not_configured" in res["reason"]


def test_model_reply_is_parsed_and_cached(room, tmp_path, monkeypatch):
    sent = []

    def complete(messages, role="super", **params):
        sent.append((messages, role, params))
        return 'Sure: {"objects": [{"label": "Coffee Mug", "box": [57, 38, 69, 54], "confidence": 0.8}]}'

    monkeypatch.setattr("arrival.models.token_factory.complete", complete)
    cache = tmp_path / "cache.json"
    res = run("mug", room, client=detect.DetectionClient(cache))
    assert res["state"] == "detected" and res["reply_sources"] == {"token_factory": 3}
    messages, role, params = sent[0]
    assert role == "omni" and params == {"temperature": 0}
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    entries = json.loads(cache.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 3 and {e["source"] for e in entries.values()} == {"token_factory"}
    monkeypatch.setattr("arrival.models.token_factory.complete", lambda *a, **k: pytest.fail("cached"))
    again = run("mug", room, client=detect.DetectionClient(cache))
    assert again["objects"][0]["position"] == res["objects"][0]["position"]


def test_unsupported_uncertain_and_unplaceable_detections_stay_explicit(room):
    good = {"label": "mug", "box": [57, 38, 69, 54], "confidence": 0.9}
    fake = FakeClient(
        {
            "f000": {
                "objects": [
                    good,
                    {"label": "other:toaster", "box": [10, 10, 30, 30], "confidence": 0.9},
                    {"label": "mug", "box": [57, 38, 69, 54], "confidence": 0.2},
                    {"label": "bowl", "box": [500, 500, 600, 600], "confidence": 0.9},
                    {"label": "cup", "box": [10, 80, 20, 90], "confidence": 0.9},  # the cabinet front, under the top
                    {"label": "plate", "box": [20, 40, 35, 55], "confidence": 0.9},  # bare counter top
                    {"label": "apple", "box": "left", "confidence": 0.9},
                ]
            }
        }
    )
    res = run("mug", room, client=fake)
    assert [o["label"] for o in res["objects"]] == ["mug"]
    assert [(r["label"], r["reason"]) for r in res["rejected"]] == [
        ("other:toaster", "unsupported_label"),
        ("mug", "low_confidence"),
        ("bowl", "bad_box"),
        ("cup", "no_support"),
        ("plate", "nothing_above_top"),
        ("apple", "bad_box"),
    ]
    assert all(r["frame"] == "f000" for r in res["rejected"])


def test_held_out_frames_and_other_rooms_are_not_read(room, tmp_path):
    shutil.copytree(FIXTURES / "mug", tmp_path / "mug")
    path = tmp_path / "mug" / rgbd.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for f in manifest["frames"]:
        f["split"] = "heldout"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    fake = FakeClient({})
    res = detect.detect_objects(tmp_path / "mug", room[1], room[0], client=fake)
    assert res["state"] == "not_run" and res["reason"] == "no train frames to read" and fake.reads == 0
    res = detect.detect_objects(FIXTURES / "mug", room[1], {**room[0], "source_room_id": "elsewhere"}, client=fake)
    assert res["state"] == "not_run" and "different rooms" in res["reason"] and fake.reads == 0


def test_two_loosely_boxed_mugs_stay_two(room, tmp_path):
    # each mug's 20 px box also holds the other mug; neither may turn into a second copy of its neighbour
    from arrival.eval.realism import render_twin

    bundle, twin = room
    mugs = [[3.12, 0.45, 0.97], [3.28, 0.45, 0.97]]
    scene = json.loads(json.dumps(twin))
    scene["objects"] = [
        {"id": f"mug_{i}", "catalog_id": "mug", "label": "mug", "size": [0.08, 0.08, 0.10], "position": p, "yaw": 0.0}
        for i, p in enumerate(mugs, 1)
    ]
    manifest = json.loads((FIXTURES / "mug" / rgbd.MANIFEST_NAME).read_text(encoding="utf-8"))
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    K = np.array([[110.0, 0, 64], [0, 110.0, 48], [0, 0, 1]])
    replies = {}
    for fr in manifest["frames"]:
        T = np.asarray(fr["camera_to_room"]).reshape(4, 4)
        rgb, depth = render_twin(scene, K, T @ np.diag([1.0, -1.0, -1.0, 1.0]), (96, 128))
        Image.fromarray(rgb).save(tmp_path / fr["rgb"])
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(tmp_path / fr["depth"])
        boxes = []
        for p in mugs:
            c = np.array(
                [
                    [p[0] + sx * 0.04, p[1] + sy * 0.04, p[2] + sz * 0.05]
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)
                ]
            )
            uv, _ = rgbd.project(c, K, T)
            boxes.append(
                {
                    "label": "mug",
                    "confidence": 0.9,
                    "box": [uv[:, 0].min() - 20, uv[:, 1].min() - 20, uv[:, 0].max() + 20, uv[:, 1].max() + 20],
                }
            )
        replies[fr["id"]] = {"objects": boxes}
    (tmp_path / rgbd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    res = detect.detect_objects(tmp_path, twin, bundle, client=FakeClient(replies), log=lambda *_: None)
    placed = sorted(o["position"][:2] for o in res["objects"])
    assert len(placed) == 2, res["objects"]
    for got, want in zip(placed, mugs):
        assert math.dist(got, want[:2]) < XY_TOL


CORNER_TABLE = {
    "id": "table_c",
    "type": "counter",
    "source_category": "table",
    "default_config_name": "dining",
    "position": [3.55, 2.55, 0.375],
    "size": [0.9, 0.9, 0.75],
    "yaw": 0.0,
}  # in the east-south corner


@pytest.mark.parametrize(
    "region,lo,hi",
    [
        ("faucet_column", (1.585, 0.10, 0.92), (1.615, 0.13, 1.13)),
        ("faucet_base", (1.575, 0.09, 0.92), (1.625, 0.14, 0.96)),
        ("east_wall_backsplash_corner", (3.92, 0.07, 0.92), (4.0, 0.15, 1.06)),
        ("fridge_backsplash_corner", (0.75, 0.07, 0.92), (0.83, 0.15, 1.06)),
        ("table_wall_corner", (3.92, 2.92, 0.75), (4.0, 3.0, 0.9)),
    ],
)
def test_corners_and_the_faucet_place_nothing(room, tmp_path, region, lo, hi):
    # where two faces meet, or at the faucet, a box finds a narrow, standing, deep group of points that is no object
    from arrival.eval.realism import look_at, render_twin

    bundle, twin = room
    if region.startswith("table"):
        twin["fixtures"].append(dict(CORNER_TABLE))
        eyes = [(3.5, 2.3, 1.3), (3.3, 2.5, 1.25), (3.6, 2.1, 1.3)]
    else:
        c = [(a + b) / 2 for a, b in zip(lo, hi)]
        eyes = [
            (c[0], c[1] + 0.75, c[2] + 0.5),
            (c[0] - 0.45, c[1] + 0.64, c[2] + 0.45),
            (c[0] + 0.45, c[1] + 0.64, c[2] + 0.45),
        ]
    target = [(a + b) / 2 for a, b in zip(lo, hi)]
    manifest = json.loads((FIXTURES / "empty" / rgbd.MANIFEST_NAME).read_text(encoding="utf-8"))
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    K = np.array([[110.0, 0, 64], [0, 110.0, 48], [0, 0, 1]])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    boxes = {}
    for fr, eye in zip(manifest["frames"], eyes):
        pose = look_at(eye, target)
        rgb, depth = render_twin(twin, K, pose, (96, 128))
        Image.fromarray(rgb).save(tmp_path / fr["rgb"])
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(tmp_path / fr["depth"])
        fr["camera_to_room"] = (pose @ np.diag([1.0, -1.0, -1.0, 1.0])).ravel().tolist()
        uv, _ = rgbd.project(corners, K, np.asarray(fr["camera_to_room"]).reshape(4, 4))
        boxes[fr["id"]] = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
    (tmp_path / rgbd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    for label in ("mug", "cup", "bottle", "book", "apple", "remote", "bowl"):
        for pad in (0, 4, 10):
            fake = FakeClient(
                {
                    f: {
                        "objects": [
                            {"label": label, "box": [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], "confidence": 0.9}
                        ]
                    }
                    for f, b in boxes.items()
                }
            )
            res = detect.detect_objects(tmp_path, twin, bundle, client=fake, log=lambda *_: None)
            assert res["objects"] == [], (region, label, pad, res["objects"])


@pytest.mark.parametrize("name", ["mug", "empty"])
def test_upload_worker_places_detections_or_labelled_defaults(tmp_path, monkeypatch, name):
    from arrival.jobs.queue import create_job
    from arrival.jobs.worker import run_once
    from arrival.web.app import make_server

    raw = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text())
    raw["objects"] = []
    out = tmp_path / "out"
    job = create_job(out / "jobs", "room.json", data=json.dumps(raw).encode())
    shutil.copytree(FIXTURES / name, out / "jobs" / job["id"] / "frames")
    monkeypatch.setattr(detect, "DEFAULT_CACHE", REPLIES)
    finished = run_once(out, episodes=1, families=["open_drawer"], people=0, res=32, log=lambda *_: None)
    assert finished["status"] == "done", finished
    twin = json.loads((Path(finished["twin_dir"]) / "twin.json").read_text())
    objects = finished["objects"]
    assert objects["counts"] == twin["provenance"]["generators"]["objects"]
    if name == "mug":
        assert objects["detection"] == "detected" and objects["counts"]["detected"] == 1
        assert objects["counts"]["default_synthetic"] == 0
        assert [o["provenance"] for o in twin["objects"]] == ["detected"]
        assert math.dist(twin["objects"][0]["position"][:2], MUG[:2]) < XY_TOL
    else:
        assert objects["detection"] == "none_found" and objects["counts"]["detected"] == 0
        assert objects["counts"]["default_synthetic"] == len(twin["objects"]) > 0
    srv = make_server(port=0, out_dir=out, scans_dir=SCANS, rate_limit=0)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/", timeout=20).read().decode()
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()
    if name == "mug":
        assert "objects: 1 detected" in page and "invented" not in page
    else:
        assert f"objects: {objects['counts']['default_synthetic']} default_synthetic (invented, not scanned)" in page


def _render(tmp_path, twin, objects, eyes, target, blocks=None):
    """Frames of `twin` holding `objects`, one per eye looking at `target`, in the fixture manifest's layout. Returns
    each frame's pixel box around each of `blocks` (default the objects; each a centre `position` and a `size`)."""
    from arrival.eval.realism import look_at, render_twin

    scene = json.loads(json.dumps(twin))
    scene["objects"] = objects
    manifest = json.loads((FIXTURES / "empty" / rgbd.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["room_id"] = twin["room_id"]
    (tmp_path / "rgb").mkdir(parents=True)
    (tmp_path / "depth").mkdir()
    K = np.array([[110.0, 0, 64], [0, 110.0, 48], [0, 0, 1]])
    boxes = {}
    for fr, eye in zip(manifest["frames"], eyes):
        pose = look_at(eye, target)
        rgb, depth = render_twin(scene, K, pose, (96, 128))
        Image.fromarray(rgb).save(tmp_path / fr["rgb"])
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(tmp_path / fr["depth"])
        fr["camera_to_room"] = (pose @ np.diag([1.0, -1.0, -1.0, 1.0])).ravel().tolist()
        boxes[fr["id"]] = []
        for o in objects if blocks is None else blocks:
            h = np.asarray(o["size"]) / 2
            c = np.array(
                [
                    [o["position"][0] + sx * h[0], o["position"][1] + sy * h[1], o["position"][2] + sz * h[2]]
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)
                ]
            )
            uv, _ = rgbd.project(c, K, np.asarray(fr["camera_to_room"]).reshape(4, 4))
            boxes[fr["id"]].append([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    (tmp_path / rgbd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return boxes


def _eyes(x, y, z, front):
    """Three viewpoints in front of x, y, z: straight on and from either side, looking down at it."""
    fx, fy = front
    return [
        (x + fx * 0.7, y + fy * 0.7, z + 0.5),
        (x + fx * 0.6 + fy * 0.5, y + fy * 0.6 - fx * 0.5, z + 0.45),
        (x + fx * 0.65 - fy * 0.55, y + fy * 0.65 + fx * 0.55, z + 0.45),
    ]


@pytest.fixture
def u_room():
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    bundle = load_room("u_kitchen_42", REPO / "tests" / "fixtures" / "scans")
    twin = compile_twin(bundle)
    twin["objects"] = []
    return bundle, twin


SPLASH_FACE = 0.012  # the scene's backsplash reaches this far over the back of a counter top (scene.counter_xml)
WALL_FACE = 0.025  # hero_kitchen's walls are 2 cm thick in the twin; the scene draws them 5 cm thick


def _against(spot, r, gap):
    """Centre x, y of an object of radius r standing `gap` from the backsplash, the east wall, or both, of hero
    counter_b (back at y 0.075, east end at the wall x 4.0), or from the backsplash of u_kitchen_42's storage_b
    (a RoomPlan storage cabinet facing -y, back at y 3.72)."""
    back, east = 0.075 + SPLASH_FACE + r + gap, 4.0 - WALL_FACE - r - gap
    return {
        "splash": (3.4, back),
        "wall": (east, 0.4),
        "corner": (east, back),
        "storage_splash": (6.9, 3.72 - SPLASH_FACE - r - gap),
    }[spot]


@pytest.mark.parametrize("gap", [0.01, 0.03])
@pytest.mark.parametrize("spot", ["splash", "wall", "corner", "storage_splash"])
def test_objects_just_in_front_of_a_backsplash_or_wall_are_placed(room, u_room, tmp_path, spot, gap):
    # the wall and backsplash strips drop points near those faces; an object 1 to 3 cm in front of them keeps enough
    bundle, twin = u_room if spot.startswith("storage") else room
    front = (0, -1) if spot.startswith("storage") else (0, 1)
    top = 0.9 if spot.startswith("storage") else 0.92
    for label in ("mug", "bottle", "bowl"):
        size = list(detect.OBJECT_SPECS[label][1])
        x, y = _against(spot, size[0] / 2, gap)
        want = [x, y, top + size[2] / 2]
        o = {"id": f"{label}_1", "catalog_id": label, "label": label, "size": size, "position": want, "yaw": 0.0}
        boxes = _render(tmp_path / label, twin, [o], _eyes(x, y, top + 0.03, front), (x, y, top + 0.03))
        for pad in (0, 6):
            fake = FakeClient(
                {
                    f: {
                        "objects": [
                            {"label": label, "box": [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], "confidence": 0.9}
                        ]
                    }
                    for f, (b,) in boxes.items()
                }
            )
            res = detect.detect_objects(tmp_path / label, twin, bundle, client=fake, log=lambda *_: None)
            assert len(res["objects"]) == 1, (label, pad, res["rejected"])
            got = res["objects"][0]["position"]
            assert math.dist(got[:2], want[:2]) < XY_TOL and abs(got[2] - want[2]) < Z_TOL, (label, pad, got, want)


@pytest.mark.parametrize(
    "region,lo,hi",
    [
        ("splash_mid", (6.85, 3.66, 0.9), (6.95, 3.74, 1.06)),
        ("splash_end", (7.36, 3.64, 0.9), (7.46, 3.74, 1.06)),
        ("splash_long", (6.6, 3.62, 0.9), (7.2, 3.74, 1.2)),
    ],
)
def test_storage_worktop_backsplash_places_nothing(u_room, tmp_path, region, lo, hi):
    # u_kitchen_42's worktops are RoomPlan storage cabinets, which the scene builds with a backsplash like a counter
    bundle, twin = u_room
    assert detect.has_backsplash(next(f for f in twin["fixtures"] if f["id"] == "storage_b"))
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    block = {"position": c, "size": [b - a for a, b in zip(lo, hi)]}
    boxes = {f: b for f, (b,) in _render(tmp_path, twin, [], _eyes(*c, (0, -1)), c, [block]).items()}
    for label in ("mug", "cup", "bottle", "book", "apple", "remote", "bowl"):
        for pad in (0, 4, 10):
            fake = FakeClient(
                {
                    f: {
                        "objects": [
                            {"label": label, "box": [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], "confidence": 0.9}
                        ]
                    }
                    for f, b in boxes.items()
                }
            )
            res = detect.detect_objects(tmp_path, twin, bundle, client=fake, log=lambda *_: None)
            assert res["objects"] == [], (region, label, pad, res["objects"])
