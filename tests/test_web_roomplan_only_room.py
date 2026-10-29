"""A phone capture lands as scans/<room>/roomplan.json with no scan.json, and `sim` builds such a room (#417). The page
listed only rooms with a scan.json: after `sim --room phone_kitchen` the index said "No rooms yet", /rooms/phone_kitchen
was a 404 and `export-site` left the room and its sim dataset out."""

import http.client
import json
import shutil
import threading
from pathlib import Path

from arrival.web.app import make_server
from arrival.web.static_export import room_data

REPO = Path(__file__).resolve().parents[1]
ROOMPLAN = REPO / "tests" / "fixtures" / "roomplan" / "kitchen_real_01.json"


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    scans, out = tmp_path / "scans", tmp_path / "out"
    (scans / "phone_kitchen").mkdir(parents=True)
    shutil.copy(ROOMPLAN, scans / "phone_kitchen" / "roomplan.json")
    meta = out / "datasets" / "phone_kitchen_sim" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {"arrival_physically_valid": True, "total_episodes": 3, "arrival_successes": 3, "arrival_attempts": 3}
        ),
        encoding="utf-8",
    )
    return scans, out


def _get(scans: Path, out: Path, path: str) -> tuple[int, str]:
    srv = make_server(port=0, out_dir=out, scans_dir=scans)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_a_roomplan_only_room_is_listed_with_its_sim_dataset(tmp_path):
    scans, out = _setup(tmp_path)
    code, page = _get(scans, out, "/")
    assert code == 200 and "No rooms yet" not in page
    assert "href='/rooms/phone_kitchen'" in page and "/download/phone_kitchen_sim" in page
    assert ">roomplan<" in page  # the geometry_source chip comes from the normalized export


def test_a_roomplan_only_room_page_shows_the_export_fixtures(tmp_path):
    scans, out = _setup(tmp_path)
    code, page = _get(scans, out, "/rooms/phone_kitchen")
    assert code == 200
    assert "roomplan.json (twin not compiled)" in page
    assert "No fixtures" not in page and "storage" in page  # kitchen_real_01 has storage cabinets


def test_a_broken_roomplan_export_still_gets_a_page_not_a_500(tmp_path):
    scans, out = _setup(tmp_path)
    (scans / "phone_kitchen" / "roomplan.json").write_text('{"walls": "nope"}', encoding="utf-8")
    assert _get(scans, out, "/")[0] == 200
    assert _get(scans, out, "/rooms/phone_kitchen")[0] == 200


def test_the_static_export_reads_the_roomplan_export(tmp_path):
    scans, out = _setup(tmp_path)
    row = room_data("phone_kitchen", out, scans)
    assert row["geometry_source"] == "roomplan" and row["fixtures"]
