"""A shard that exits 0 without the dataset's meta files is a fetch failure.

`run_fanout` copies the shard folder, then requires `meta/info.json` and
`meta/episodes.jsonl`. That check used to run after `StatusRecorder.fetch_result`
had already treated the copy as complete, so `arrival fanout`'s `cancel_open`
recorded `failed_stage` `marker` and the reason "the fan-out stopped before this
result was stored". The missing path was dropped. A non-zero exit stays `run`;
a missing shard folder is already `fetch`.
"""

import json
import sys

from arrival.jobs.fanout import FanoutConfig, LocalBackend, run_fanout
from arrival.jobs.fanout_status import StatusRecorder, status_key
from arrival.jobs.shards import ShardError

CFG = FanoutConfig("r1", 1, "hero_kitchen", "open_drawer", 1, 0, 32, 0, "scans")
QUIET = lambda *_: None  # noqa: E731


def _command(kind):
    script = (
        "import pathlib, sys\n"
        "out, shard, kind = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]\n"
        "meta = out / 'shards' / shard / 'datasets' / 'hero_kitchen_sim' / 'meta'\n"
        "meta.mkdir(parents=True)\n"
        "if kind == 'info':\n"
        "    (meta / 'info.json').write_text('{}')\n"
    )

    def command(cfg, i, out):
        return [sys.executable, "-c", script, str(out), f"{i}of{cfg.shards}", kind]

    return command


def test_exit_zero_without_meta_files_is_recorded_as_fetch(tmp_path, kind, missing):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch", command=_command(kind))
    rec = StatusRecorder(be)
    caught = None
    try:
        run_fanout(rec, CFG, tmp_path / "out", poll_seconds=0, max_attempts=2, log=QUIET)
    except ShardError as e:
        caught = e
    finally:
        rec.cancel_open()  # arrival fanout does this on the way out; that is what writes the wrong stage
    assert caught is not None and missing in str(caught)
    [attempt] = json.loads(be.get_object(status_key(CFG, 0)))["attempts"]
    assert (attempt["outcome"], attempt["failed_stage"]) == ("failed", "fetch")
    assert missing in attempt["reason"] and "stopped before" not in attempt["reason"]
    assert attempt["states"][-1]["state"] == "exited 0"
    assert be.submissions == [(0, 1)]  # a fetch failure is not a retryable shard exit


def pytest_generate_tests(metafunc):
    if "kind" in metafunc.fixturenames:
        metafunc.parametrize("kind,missing", [("empty", "info.json"), ("info", "episodes.jsonl")])
