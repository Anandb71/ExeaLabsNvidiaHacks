"""arrival cost lists the scripted placeholder report `arrival demo` writes, but never compares it with teleop."""

import json
from pathlib import Path

from arrival.eval import cost
from arrival.eval.cost import build_report, write_report

BASELINE = {
    "task": "all_tasks",
    "room": "hero_kitchen",
    "n_success": 10,
    "operator_seconds_total": 1500,
    "setup_assumptions": ["one operator"],
    "cost_basis_usd": 50,
}


def _w(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    # what `arrival sim` writes: physics-checked, 10 successes in 120 s
    _w(
        out / "reports" / "hero_kitchen_sim.json",
        {
            "room_id": "hero_kitchen",
            "generator": "mujoco_scripted_ik",
            "physics_checked_success": True,
            "attempts": 12,
            "successes": 10,
            "seconds": 120.0,
            "human_contacts": 1,
            "families": {"counter_to_sink": {"attempts": 12, "successes": 10, "human_contacts": 1, "seconds": 120.0}},
        },
    )
    # what `arrival demo` writes for the same room: scripted placeholder motion, 4 kept attempts in 0.25 s
    write_report(build_report("hero_kitchen", 4, 4, 0.25, ["counter_to_counter", "counter_to_sink"]), out / "reports")
    return out


def test_room_level_comparison_counts_only_the_physics_checked_report(tmp_path):
    base = _w(tmp_path / "b.json", BASELINE)
    c = cost.build(_out(tmp_path), baseline_path=base)["comparison"]
    assert c["generated"]["scope"] == "room"
    assert c["generated"]["n_success"] == 10
    assert c["generated"]["seconds_per_success"] == 12.0
    assert c["seconds_ratio_teleop_over_generated"] == 12.5


def test_the_placeholder_report_is_listed_and_labelled(tmp_path):
    rep = cost.build(_out(tmp_path))
    marked = [(r["source"], r["family"], r["successes"], r["seconds"]) for r in rep["rows"] if r.get("placeholder")]
    assert marked == [("report", None, 4, 0.25)]
    text = cost.markdown(rep)
    assert "| report (placeholder) | hero_kitchen | all | 4 | 4 |" in text
    assert "| report | hero_kitchen | all | 12 | 10 |" in text
