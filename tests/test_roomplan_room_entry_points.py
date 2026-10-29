"""A phone capture lands as scans/<room>/roomplan.json only. #417 taught sim, eval and fanout to build such a room;
teleop, both retarget generators and scripts/check_layout.py still read scan.json only and said "no scan for room".
They now share arrival.twin.compile.room_twin, which builds the same twin `sim` does."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from arrival import cli  # noqa: E402
from arrival.twin.compile import room_twin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "tests" / "fixtures" / "roomplan" / "kitchen_real_01.json"


@pytest.fixture
def scans(tmp_path):
    (tmp_path / "phone_kitchen").mkdir()
    shutil.copy(EXPORT, tmp_path / "phone_kitchen" / "roomplan.json")
    return tmp_path


def test_room_twin_builds_a_phone_capture_like_sim(scans, tmp_path):
    twin = room_twin("phone_kitchen", scans)
    assert twin == cli._load_twin("phone_kitchen", tmp_path / "out", scans)  # what sim runs on
    assert twin["room_id"] == "phone_kitchen" and twin["objects"]
    with pytest.raises(FileNotFoundError):
        room_twin("no_such_room", scans)


def test_room_twin_still_prefers_scan_json():
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    assert room_twin("hero_kitchen", ROOT / "scans") == compile_twin(load_room("hero_kitchen", ROOT / "scans"))


def test_check_layout_checks_a_phone_capture(scans, capsys):
    spec = importlib.util.spec_from_file_location("check_layout", ROOT / "scripts" / "check_layout.py")
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    assert cl.main(["--room", "phone_kitchen", "--scans", str(scans), "--json"]) == 0  # was exit 2, "no scan for room"
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["fixtures"] == 6


@pytest.mark.parametrize(
    "module, target", [("arrival.generate.se3_retarget", "generate"), ("arrival.generate.retarget", "retarget")]
)
def test_retarget_generators_take_a_phone_capture(scans, tmp_path, monkeypatch, module, target):
    mod = importlib.import_module(module)
    seen = {}

    def fake(twin, *args, **kwargs):
        seen["room"] = twin["room_id"]
        return {"report_data": {"attempts": 0}}

    monkeypatch.setattr(mod, target, fake)
    assert mod.main(["--room", "phone_kitchen", "--scans", str(scans), "--out", str(tmp_path / "out")]) == 0
    assert seen["room"] == "phone_kitchen"


def test_teleop_takes_a_phone_capture(scans, tmp_path, monkeypatch):
    from arrival.sim import teleop

    seen = {}

    class Window:
        def attach(self, env):
            return self

        def shutdown(self):
            pass

    def fake_run(twin, frames, *args, **kwargs):
        seen["room"] = twin["room_id"]
        return {"room": twin["room_id"]}

    monkeypatch.setattr(teleop, "TeleopWindow", Window)
    monkeypatch.setattr(teleop, "run", fake_run)
    assert (
        teleop.main(
            [
                "--room",
                "phone_kitchen",
                "--scans",
                str(scans),
                "--out",
                str(tmp_path / "out"),
                "--operator-usd-per-hour",
                "2",
            ]
        )
        == 0
    )
    assert seen["room"] == "phone_kitchen"
