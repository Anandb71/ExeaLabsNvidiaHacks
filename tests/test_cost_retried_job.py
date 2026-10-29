"""arrival cost counts the wall time of every attempt of a retried job, not only the attempt that finished it."""

import json
import shutil

import pytest

from arrival.eval import cost
from arrival.jobs.worker import record_usage

USAGE = {
    "wall_seconds": 1500.0,
    "episodes_attempted": 40,
    "episodes_succeeded": 31,
    "frames": 12400,
    "upload_bytes": 52000,
    "zip_bytes": None,
    "dataset_bytes": 9000000,
    "measured_on": "local_cpu",
}


def _retried_job(out, first_wall=1500.0):
    """A job whose first attempt failed on a full disk and went back to the queue, then finished on the second: the
    worker leaves one ledger line per attempt, and status.json holds the second attempt."""
    first = {
        "id": "job_x",
        "room_id": "kitchen_a",
        "status": "failed",
        "started_at": 1000,
        "finished_at": 2500,
        "job_attempts": 1,
        "attempts": 40,
        "successes": 31,
        "usage": {**USAGE, "wall_seconds": first_wall},
    }
    second = {
        **first,
        "status": "done",
        "started_at": 2600,
        "finished_at": 4200,
        "job_attempts": 2,
        "successes": 32,
        "retry_errors": ["OSError: [Errno 28] No space left on device"],
        "usage": {
            **USAGE,
            "wall_seconds": 1600.0,
            "episodes_succeeded": 32,
            "frames": 12800,
            "zip_bytes": 4000000,
            "dataset_bytes": 9100000,
        },
    }
    record_usage(out, first)
    record_usage(out, second)
    (out / "jobs" / "job_x").mkdir(parents=True)
    (out / "jobs" / "job_x" / "status.json").write_text(json.dumps(second))


@pytest.mark.parametrize("gone", [False, True])
def test_a_retried_job_counts_the_time_of_every_attempt(tmp_path, gone):
    out = tmp_path / "out"
    _retried_job(out)
    if gone:  # `arrival gc` deleted the job folder; the ledger keeps the job
        shutil.rmtree(out / "jobs")
    rep = cost.build(out)
    (row,) = [r for r in rep["rows"] if r["source"] == "job"]
    assert row["seconds"] == 3100.0  # 1500 s failed + 1600 s done, the status.json line not counted twice
    assert (row["attempts"], row["successes"]) == (40, 32)  # episodes of the attempt that finished the job
    assert row["seconds_per_demo"] == 96.875 and row["demos_per_hour"] == 37.2
    m = rep["metering"]
    assert m["jobs"] == 1 and m["wall_seconds"] == 3100.0
    assert (m["episodes_succeeded"], m["frames"], m["zip_bytes"]) == (32, 12800, 4000000)
    assert m["upload_bytes"] == 52000  # one upload, read again by each attempt
    assert "wall_seconds 3100.0" in cost.markdown(rep)


@pytest.mark.parametrize("bad", [-5, "1500", None])
def test_an_earlier_attempt_with_bad_wall_seconds(tmp_path, bad):
    out = tmp_path / "out"
    _retried_job(out, first_wall=bad)
    if bad is None:  # not measured: the attempts that were measured still add up
        assert cost.metering_totals(out)["wall_seconds"] == 1600.0
    else:
        with pytest.raises(cost.CostError, match="wall_seconds"):
            cost.build(out)
