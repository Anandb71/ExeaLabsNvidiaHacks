"""A fan-out that stops on a failed shard also stops its other local shard processes.

run_fanout raises when one shard fails its last attempt. `arrival fanout` cancelled the jobs still open only for the
Nebius backend, so the local backend's other `arrival sim` children kept running after the command exited, and the
rerun its message asks for started a second process for the same shard in the same scratch folder.
"""

import json
import sys

import pytest

from arrival import cli
from arrival.jobs import fanout
from arrival.jobs.fanout import FanoutConfig, LocalBackend, run_fanout
from arrival.jobs.fanout_status import StatusRecorder, status_key
from arrival.jobs.shards import ShardError

FAIL = [sys.executable, "-c", "import sys; sys.exit(3)"]
SLOW = [sys.executable, "-c", "import time; time.sleep(120)"]


class OneShardFails(LocalBackend):
    """Shard 0 exits non-zero at once; every other shard runs until it is stopped."""

    def __init__(self, store, scratch):
        super().__init__(store, scratch, command=lambda cfg, i, out: FAIL if i == 0 else SLOW)

    def procs(self):
        return [proc for proc, *_ in self._jobs.values()]


@pytest.fixture
def made():
    backends = []
    yield backends
    for be in backends:  # never leave a sleeper behind, even when an assertion failed
        for proc in be.procs():
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_cancel_open_stops_the_shards_still_running(tmp_path, made):
    be = OneShardFails(tmp_path / "store", tmp_path / "scratch")
    made.append(be)
    rec, cfg = StatusRecorder(be), FanoutConfig("r", 2)
    with pytest.raises(ShardError, match="failed 3 times"):
        run_fanout(rec, cfg, tmp_path / "out", poll_seconds=0.01, log=lambda *_: None)
    assert [p.poll() for p in be.procs() if p.args == SLOW] == [None]  # run_fanout itself stops nothing
    assert rec.cancel_open() == []
    assert all(p.poll() is not None for p in be.procs())
    last = json.loads(be.get_object(status_key(cfg, 1)))["attempts"][-1]
    assert (last["outcome"], last["failed_stage"]) == ("failed", "run")


def test_fanout_command_leaves_no_shard_running(tmp_path, made, monkeypatch, capsys):
    monkeypatch.setattr(
        fanout, "LocalBackend", lambda store, scratch: made.append(OneShardFails(store, scratch)) or made[-1]
    )
    rc = cli.main(["fanout", "--room", "hero_kitchen", "--shards", "2", "--episodes", "2", "--out", str(tmp_path)])
    assert rc == cli.EXIT_USAGE and "shard 0/2 failed 3 times" in capsys.readouterr().err
    assert made and len(made[0].procs()) == 4  # three tries of shard 0, one of shard 1
    assert all(p.poll() is not None for p in made[0].procs())
