"""DART (`arrival sim --dart`): the arm is driven off the plan, the dataset keeps the clean plan as the action."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import SimEnv, _grip_events, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.tasks import finish_task, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def _rollout(twin, exec_noise):
    task = finish_task(make_tasks(twin, ["counter_to_sink"], 1, seed=3, people=0, log=lambda *_: None)[0])
    with SimEnv(twin, task, res=32) as env:
        return run_episode(env, ScriptedExpert(), task, exec_noise=exec_noise)


def test_off_by_default_changes_nothing(twin):
    a, b = _rollout(twin, 0.0), _rollout(twin, 0.0)
    assert np.array_equal(a["actions"], b["actions"]) and np.array_equal(a["states"], b["states"])


def test_noise_moves_the_arm_but_not_the_recorded_action(twin):
    clean, noisy = _rollout(twin, 0.0), _rollout(twin, 0.04)
    assert np.array_equal(clean["actions"], noisy["actions"])  # the label is the clean plan
    arm_gap = np.abs(noisy["states"][:, :7] - clean["states"][:, :7])
    assert arm_gap.max() > 0.01  # the arm really was somewhere else
    assert arm_gap.max() < 0.3  # a perturbation, not a different motion
    again = _rollout(twin, 0.04)
    assert np.array_equal(again["states"], noisy["states"])  # seeded from the task: repeatable


def test_gripper_targets_are_not_perturbed_and_events_are_found(twin):
    plan = [np.array([0, 0, 0, 0, 0, 0, 0, g]) for g in (0.05, 0.05, 0.0, 0.0, 0.05)]
    assert list(_grip_events(plan)) == [1, 3]
    assert len(_grip_events(None)) == 0
    clean, noisy = _rollout(twin, 0.0), _rollout(twin, 0.04)
    assert np.array_equal(clean["actions"][:, 7], noisy["actions"][:, 7])


def test_journal_resumes_old_runs_as_no_noise_and_refuses_another_level(tmp_path):
    import json

    from arrival.jobs.shards import FINGERPRINT_KEYS, Journal, ShardError

    old = {k: None for k in FINGERPRINT_KEYS if k != "exec_noise"}  # a journal written before DART existed
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps({"fingerprint": old}) + "\n")
    now = {k: None for k in FINGERPRINT_KEYS}
    assert Journal(path).start({**now, "exec_noise": 0.0}, resume=True) == []
    with pytest.raises(ShardError, match="exec_noise"):
        Journal(path).start({**now, "exec_noise": 0.04}, resume=True)


def test_merged_shards_keep_the_noise_level_and_refuse_a_mix(twin, tmp_path):
    import json

    from arrival.jobs.shards import ShardError, merge_shards
    from arrival.sim.collect import collect

    def shards(root, noise_for):
        for i in range(2):
            collect(
                twin,
                ["counter_to_sink"],
                4,
                root,
                people=0,
                res=32,
                seed=2,
                shard=(i, 2),
                exec_noise=noise_for(i),
                log=lambda *_: None,
            )

    shards(tmp_path / "a", lambda i: 0.02)
    out = merge_shards([str(tmp_path / "a")], tmp_path / "merged")
    info = json.loads((out["dataset"] / "meta" / "info.json").read_text())
    assert info["arrival_exec_noise_rad"] == 0.02
    assert info["arrival_provenance"]["run"]["exec_noise"] == 0.02  # the shards' provenance survives the merge
    for f in (tmp_path / "a" / "shards").glob(
        "*/datasets/*/meta/info.json"
    ):  # shards written before the info key existed
        d = json.loads(f.read_text())
        d.pop("arrival_exec_noise_rad")
        f.write_text(json.dumps(d))
    legacy = merge_shards([str(tmp_path / "a")], tmp_path / "merged_legacy")
    assert legacy["info"]["arrival_exec_noise_rad"] == 0.02  # recovered from the provenance block
    shards(tmp_path / "b", lambda i: 0.02 if i == 0 else 0.0)
    with pytest.raises(ShardError, match="arrival_exec_noise_rad"):
        merge_shards([str(tmp_path / "b")], tmp_path / "merged_b")
    shards(tmp_path / "c", lambda i: 0.0)  # a clean run is unchanged: no key at all
    clean = merge_shards([str(tmp_path / "c")], tmp_path / "merged_c")
    assert "arrival_exec_noise_rad" not in clean["info"]


@pytest.mark.parametrize("dart", ["-0.02", "nan", "inf"])
def test_sim_refuses_a_dart_that_is_not_a_finite_number_at_least_0(tmp_path, monkeypatch, capsys, dart):
    """-0.02 and nan ran with no noise and still wrote that value to info.json (nan as NaN, which is not JSON)."""
    from arrival import cli

    monkeypatch.setattr("arrival.sim.collect.collect", lambda *a, **k: pytest.fail("collect ran"))
    assert cli.main(["sim", "--room", "hero_kitchen", "--dart", dart, "--out", str(tmp_path)]) == cli.EXIT_USAGE
    assert "--dart must be a finite number >= 0" in capsys.readouterr().err
