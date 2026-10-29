"""scripts/render_rooms.py names a roomplan.json given as a file after its folder, as it names the folder itself.
By the file's stem every scans/<room>/roomplan.json (where intake_scan.py lands a scan) was out/renders/roomplan_top.png,
so `render_rooms.py scans/*/roomplan.json` kept only the last room's plan and printed the same path for each."""

import importlib.util
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "roomplan"


def _render_rooms():
    spec = importlib.util.spec_from_file_location("render_rooms", REPO / "scripts" / "render_rooms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scans(tmp_path) -> Path:
    for room, fixture in (("kitchen_a", "kitchen_real_01.json"), ("office_b", "office.json")):
        (tmp_path / "scans" / room).mkdir(parents=True)
        shutil.copy(FIXTURES / fixture, tmp_path / "scans" / room / "roomplan.json")
    return tmp_path / "scans"


def test_each_roomplan_json_gets_its_own_render(tmp_path, capsys):
    scans, out = _scans(tmp_path), tmp_path / "renders"
    files = [str(scans / "kitchen_a" / "roomplan.json"), str(scans / "office_b" / "roomplan.json")]
    assert _render_rooms().main([*files, "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["kitchen_a_roomplan_top.png", "office_b_roomplan_top.png"]
    assert capsys.readouterr().out.split() == [
        str(out / "kitchen_a_roomplan_top.png"),
        str(out / "office_b_roomplan_top.png"),
    ]


def test_the_file_and_its_folder_name_the_room_alike(tmp_path):
    scans, mod = _scans(tmp_path), _render_rooms()
    ((by_file, twin),) = mod._twins([scans / "kitchen_a" / "roomplan.json"])
    ((by_folder, _),) = mod._twins([scans / "kitchen_a"])
    assert by_file == by_folder == "kitchen_a_roomplan"
    assert twin["room_id"] == "kitchen_a"


def test_another_export_name_keeps_its_stem(tmp_path):
    ((name, twin),) = _render_rooms()._twins([FIXTURES / "office.json"])
    assert name == "office" and twin["room_id"] == "office"
