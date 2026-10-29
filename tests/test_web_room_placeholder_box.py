"""The room page's placeholder box ("arrival_physically_valid: false ... Do not fine-tune on it") is about the scripted
placeholder in out/datasets/<room>. A room whose only dataset is the physics-checked <room>_sim got the box too, right
above its own headline saying "physically valid: true"."""

import http.client
import json
import threading
from pathlib import Path

import pytest

from arrival.web.app import make_server

REPO = Path(__file__).resolve().parents[1]
BOX = "Do not fine-tune on it while this flag is false"


def _info(out: Path, name: str, **info) -> None:
    meta = out / "datasets" / name / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _room_page(out: Path, path: str = "/rooms/hero_kitchen") -> str:
    srv = make_server(port=0, out_dir=out, scans_dir=REPO / "scans")
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        conn.request("GET", path)
        resp = conn.getresponse()
        assert resp.status == 200
        return resp.read().decode("utf-8")
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


SIM = dict(
    arrival_physically_valid=True,
    arrival_generator="mujoco_scripted_ik",
    total_episodes=2,
    arrival_successes=2,
    arrival_attempts=2,
)
DEMO = dict(arrival_physically_valid=False, arrival_generator="scripted_retarget", total_episodes=4)


def test_a_room_with_only_the_sim_dataset_gets_no_placeholder_box(tmp_path):
    _info(tmp_path, "hero_kitchen_sim", **SIM)
    page = _room_page(tmp_path)
    assert BOX not in page and "arrival_physically_valid: false" not in page
    assert "physically valid: true" in page


@pytest.mark.parametrize("names", [("hero_kitchen",), ("hero_kitchen", "hero_kitchen_sim")])
def test_the_box_stays_while_the_placeholder_dataset_exists(tmp_path, names):
    for name in names:
        _info(tmp_path, name, **(SIM if name.endswith("_sim") else DEMO))
    page = _room_page(tmp_path)
    assert BOX in page and "arrival_physically_valid: false" in page


# The front page showed the same placeholder note to every visitor, above physics-checked sim and upload datasets.
def test_the_front_page_has_no_placeholder_note_when_no_room_has_the_placeholder(tmp_path):
    _info(tmp_path, "hero_kitchen_sim", **SIM)
    page = _room_page(tmp_path, "/")
    assert "not physically valid" not in page


def test_the_front_page_keeps_the_note_while_a_room_has_the_placeholder(tmp_path):
    _info(tmp_path, "hero_kitchen", **DEMO)
    _info(tmp_path, "hero_kitchen_sim", **SIM)
    page = _room_page(tmp_path, "/")
    assert "not physically valid" in page
