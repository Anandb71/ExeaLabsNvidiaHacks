"""sim --resume refuses a journal written in another twin (the scan was edited after the kill) instead of keeping the
old room's episodes next to the new room's."""

import json

import pytest

mujoco = pytest.importorskip("mujoco")

from conftest import SCANS  # noqa: E402

from arrival.jobs.shards import ShardError  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import collect as collect_mod  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

RUN = dict(families=["open_drawer"], episodes=2, people=0, res=32, seed=0, log=lambda *_: None)


class Killed(BaseException):
    """A kill during a task: not an Exception, so collect does not count it as a skipped task."""


def test_resume_refuses_progress_from_another_twin_and_takes_a_recompile(tmp_path, monkeypatch):
    twin = compile_twin(load_room("hero_kitchen", SCANS))
    real, calls = collect_mod.run_episode, []

    def killed(env, policy, task, **kw):
        calls.append(task["task_id"])
        if len(calls) > 1:
            raise Killed
        return real(env, policy, task, **kw)

    monkeypatch.setattr(collect_mod, "run_episode", killed)
    with pytest.raises(Killed):
        collect_mod.collect(twin, out_dir=tmp_path, **RUN)
    journal = tmp_path / "progress" / "hero_kitchen_sim.jsonl"
    ds = tmp_path / "datasets" / "hero_kitchen_sim"
    saved = journal.read_text()
    assert len(saved.splitlines()) == 2  # the fingerprint and the one task that finished

    edited = json.loads(json.dumps(twin))  # the scan was edited after the kill: counter_b moved 5 cm along its wall
    next(f for f in edited["fixtures"] if f["id"] == "counter_b")["position"][0] -= 0.05
    edited["provenance"]["input"]["file_sha256"] = "0" * 64
    with pytest.raises(ShardError, match="different twin"):
        collect_mod.collect(edited, out_dir=tmp_path, resume=True, **RUN)
    assert len(calls) == 2  # no task ran in the edited room
    assert journal.read_text() == saved
    assert len(list(ds.glob("data/chunk-*/*.parquet"))) == 1  # the kept episode is still there

    # the same scan compiled again (another process, another code checkout) is the same twin: the resume skips the
    # finished task and goes on with the next one
    again = compile_twin(load_room("hero_kitchen", SCANS))
    again["provenance"]["code"] = {"sha": "f" * 40, "dirty": True, "source": "git"}
    with pytest.raises(Killed):
        collect_mod.collect(again, out_dir=tmp_path, resume=True, **RUN)
    assert calls == ["sim_open_drawer_0000", "sim_open_drawer_0001", "sim_open_drawer_0001"]
