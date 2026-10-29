"""A scan or frames file named without Latin letters (Кухня.json, 厨房.json, кадры.zip) is still a .json / .zip upload.

safe_filename replaced the whole stem with "_" and then stripped it together with the dot, so the name came out as
"json" and POST /jobs answered 415 "the scan must be a .json file" for a valid RoomPlan export.
"""

import http.client
import io
import json
import threading
import zipfile
from pathlib import Path

import pytest

from arrival.web import hardening as hd
from arrival.web.app import make_server

REPO = Path(__file__).resolve().parents[1]
SCAN = b'{"a": 1}'


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Кухня.json", "upload.json"),
        ("厨房.json", "upload.json"),
        ("キッチン.JSON", "upload.JSON"),
        ("مطبخ.json", "upload.json"),
        ("кадры.zip", "upload.zip"),
        (".json", "upload.json"),
    ],
)
def test_name_without_latin_letters_keeps_its_extension(raw, expected):
    assert hd.safe_filename(raw) == expected
    assert hd.safe_filename(raw, "job_0123abcd") == "job_0123abcd" + expected[len("upload") :]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Küche.json", "K_che.json"),
        ("....", "upload.bin"),
        ("Кухня", "upload.bin"),
        ("..hidden.json", "hidden.json"),
    ],
)
def test_names_with_latin_letters_are_unchanged(raw, expected):
    assert hd.safe_filename(raw) == expected


@pytest.fixture
def server(tmp_path):
    out = tmp_path / "out"
    srv = make_server(port=0, out_dir=out, scans_dir=REPO / "scans", rate_limit=0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        yield srv, out
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _post(srv, parts):
    body = b""
    for name, filename, payload in parts:
        body += (
            (f'--bnd7\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n\r\n').encode()
            + payload
            + b"\r\n"
        )
    body += b"--bnd7--\r\n"
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request("POST", "/jobs", body=body, headers={"Content-Type": "multipart/form-data; boundary=bnd7"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw) if raw else {}


def test_upload_named_in_cyrillic_is_queued(server):
    srv, out = server
    code, data = _post(srv, [("scan", "Кухня.json", SCAN)])
    assert code == 202, data
    status = json.loads((out / "jobs" / data["id"] / "status.json").read_text(encoding="utf-8"))
    assert status["filename"] == "upload.json"


def test_frames_zip_named_in_cyrillic_is_accepted(server, tmp_path):
    from arrival.scan.rgbd import write_synthetic_bundle

    root = write_synthetic_bundle(tmp_path / "bundle")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(root).as_posix())
    srv, out = server
    code, data = _post(srv, [("scan", "kitchen.json", SCAN), ("frames", "кадры.zip", buf.getvalue())])
    assert code == 202, data
    assert (out / "jobs" / data["id"] / "frames").is_dir()
