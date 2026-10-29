"""G7 / issue #62: recorded teleop demos as SE(3) retargeting seeds (arrival.generate.teleop_seeds)."""

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.export.validate import validate  # noqa: E402
from arrival.generate.se3_retarget import generate  # noqa: E402
from arrival.generate.teleop_seeds import GENERATOR, TeleopReplay, load_demos, teleop_sources  # noqa: E402
from arrival.sim.collect import SimEnv, run_episode  # noqa: E402
from arrival.sim.export import SimDatasetWriter  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.robot import FPS  # noqa: E402
from arrival.sim.tasks import finish_task, make_tasks  # noqa: E402

RES = 16


@pytest.fixture(scope="module")
def twin():
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    return compile_twin(load_room("hero_kitchen", "scans"))


@pytest.fixture(scope="module")
def demo(twin):
    """One successful counter_to_sink episode, standing in for a human demo (the P3 archive is not in the repo)."""
    task = finish_task(make_tasks(twin, ["counter_to_sink"], 3, seed=0, people=0, log=lambda *_: None)[0])
    with SimEnv(twin, task, res=RES, cameras=()) as env:
        ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"]
    return task, ep


def write_run(root, task, ep, actions=None, instruction=None):
    """A folder laid out like arrival.sim.teleop's --record output and the P3 archive."""
    actions = ep["actions"] if actions is None else actions
    n = len(actions)
    writer = SimDatasetWriter(
        root / "datasets" / "hero_kitchen_teleop", "hero_kitchen", FPS, RES, ("ego_view", "wrist")
    )
    writer.add_episode(
        {
            "states": ep["states"],
            "actions": actions,
            "detail": ep["detail"],
            "safety": ep["safety"],
            "videos": {c: np.zeros((n, RES, RES, 3), np.uint8) for c in ("ego_view", "wrist")},
            "people_xy": [[] for _ in range(n)],
        },
        task,
    )
    writer.finish(2)
    (root / "reports").mkdir(parents=True)
    rows = [
        {"attempt": 0, "task_id": task["task_id"], "instruction": instruction or task["instruction"], "success": True},
        {"attempt": 1, "task_id": "sim_counter_to_sink_0001", "instruction": "x", "success": False},
    ]
    (root / "reports" / "hero_kitchen_teleop_attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "session.json").write_text(json.dumps({"arguments": ["--successes", 1, "--seed", "0", "--people", "0"]}))
    return root


def test_a_recorded_demo_becomes_a_teleop_source_and_retargets_in_physics(twin, demo, tmp_path):
    task, ep = demo
    sources, rows = teleop_sources(twin, write_run(tmp_path / "run", task, ep), log=lambda *_: None)
    assert [r["outcome"] for r in rows] == ["source"]
    (src,) = sources
    assert src["source_generator"] == GENERATOR and src["teleop"] == {"episode_index": 0, "attempt": 0}
    assert {s["segment"] for s in src["samples"]} == {"object", "sink"}
    r = generate(
        twin,
        tmp_path / "out",
        sources=[src, src],
        res=32,
        offsets=[(0.03, -0.01, 0.08), (-0.03, 0.01, -0.08)],
        log=lambda *_: None,
    )
    report = r["report_data"]
    assert report["successes"] == report["attempts"] == 2, report["rows"]
    assert report["source_generator"] == [GENERATOR]
    assert report["per_source"][0]["attempts"] == 2  # the same seed twice: both rows count against it
    assert validate(r["dataset"]) == []
    info = json.loads((r["dataset"] / "meta/info.json").read_text())
    assert info["arrival_source_ref"] == "meta/se3_sources.json"
    assert json.loads((r["dataset"] / "meta/se3_sources.json").read_text())[0]["source_id"] == src["source_id"]
    # `arrival cost` files the run under the room and family, so it is compared with a teleop baseline (C3)
    from arrival.eval.cost import collect_rows

    row = next(x for x in collect_rows(tmp_path / "out") if x["family"] == "counter_to_sink")
    assert row["room"] == "hero_kitchen" and row["successes"] == 2 and row["seconds_per_demo"] > 0


def test_a_demo_that_no_longer_passes_physics_is_skipped_not_used(twin, demo, tmp_path):
    task, ep = demo
    open_hand = np.array(ep["actions"], dtype=float)
    open_hand[:, 7] = 0.05  # the hand never closes, so the object is never carried to the sink
    sources, rows = teleop_sources(twin, write_run(tmp_path, task, ep, actions=open_hand), log=lambda *_: None)
    assert sources == [] and rows[0]["outcome"] in ("skip_replay_fail", "skip_never_moved")


def test_a_log_from_another_run_is_refused(demo, tmp_path):
    task, ep = demo
    with pytest.raises(ValueError, match="different runs"):
        load_demos(write_run(tmp_path, task, ep, instruction="Put the plate in the sink."), "hero_kitchen")


def test_float32_open_command_is_clipped_to_the_finger_range(twin, demo):
    task, _ = demo
    a = np.zeros(8, np.float32)
    a[7] = 0.05
    policy = TeleopReplay([a])
    with SimEnv(twin, task, res=RES, cameras=()) as env:
        policy.reset(env, task)
        out = policy.act({})
        assert out[7] == env.finger_open and len(out) == 11
