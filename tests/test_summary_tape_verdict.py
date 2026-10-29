"""The summary table's tape cell says when the twin-vs-tape comparison failed, and why.

It printed the largest error over the dimensions the twin had, so a room whose scan has no sink showed
"0.0 cm max" while summary.json said ok: false and sink_width_m: null, and a counter 6 cm off showed "6.0 cm max"
with no word that the tolerance is 5 cm.
"""

import json
import shutil

from conftest import REPO, SCANS

from arrival.eval import summary

MEASURED = REPO / "tests" / "fixtures" / "measurements" / "hero_kitchen.json"


def _cell(tape: dict) -> str:
    row = {
        "room": "r",
        "geometry_source": "roomplan",
        "tape": tape,
        "stages": {s: {"status": "not_run", "detail": ""} for s in summary.STAGES},
    }
    md = summary.to_markdown({"settings": {"episodes_per_room": 1, "res": 64, "people": 0}, "rooms": [row]})
    return next(ln for ln in md.splitlines() if ln.startswith("| r |")).split(" | ")[8]


def _no_sink_room(tmp_path, monkeypatch) -> dict:
    import arrival.sim.tasks as tasks

    monkeypatch.setattr(tasks, "make_tasks", lambda *a, **k: [])  # the tape column is the point; skip the sim
    scan = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text())
    scan["fixtures"] = [f for f in scan["fixtures"] if f["category"] != "sink"]
    room = tmp_path / "scans" / "no_sink"
    room.mkdir(parents=True)
    (room / "scan.json").write_text(json.dumps(scan))
    shutil.copy(MEASURED, room / "measurements.json")
    return summary.run_room("no_sink", tmp_path / "out", tmp_path / "scans")


def test_a_dimension_missing_from_the_twin_fails_the_cell(tmp_path, monkeypatch):
    tape = _no_sink_room(tmp_path, monkeypatch)["tape"]
    assert tape["ok"] is False and tape["dimensions"]["sink_width_m"] is None
    assert _cell(tape) == "0.0 cm max, FAIL: sink_width_m not in the twin (authored, not tape)"


def test_an_error_over_the_tolerance_fails_the_cell():
    tape = {
        "max_abs_error_m": 0.06,
        "tolerance_m": 0.05,
        "ok": False,
        "tape_measured": True,
        "dimensions": {"counter_length_m": 0.06, "counter_height_m": 0.0, "sink_width_m": 0.01},
    }
    assert _cell(tape) == "6.0 cm max, FAIL: over the 5.0 cm tolerance"


def test_both_reasons_are_given():
    tape = {
        "max_abs_error_m": 0.08,
        "tolerance_m": 0.05,
        "ok": False,
        "tape_measured": True,
        "dimensions": {"counter_length_m": 0.08, "counter_height_m": 0.0, "sink_width_m": None},
    }
    assert _cell(tape) == "8.0 cm max, FAIL: sink_width_m not in the twin, over the 5.0 cm tolerance"


def test_a_passing_comparison_reads_as_before():
    tape = {
        "max_abs_error_m": 0.012,
        "tolerance_m": 0.05,
        "ok": True,
        "tape_measured": False,
        "dimensions": {"counter_length_m": 0.012, "counter_height_m": 0.0, "sink_width_m": 0.0},
    }
    assert _cell(tape) == "1.2 cm max (authored, not tape)"
