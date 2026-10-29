import pytest

from arrival.jobs.shards import Journal, ShardError

FP = {
    "room": "r",
    "families": ["a"],
    "episodes": 1,
    "seed": 0,
    "people": 0,
    "policy": "scripted",
    "planner": "rule",
    "split": "train",
    "instructions": False,
    "base": "fixed",
    "res": 32,
    "shard": [0, 1],
}


@pytest.mark.parametrize("record", [[], {"task_id": "t0"}, {"task_id": 0, "outcome": "ok"}])
def test_resume_rejects_malformed_complete_record_without_expected_tasks(tmp_path, record):
    journal = Journal(tmp_path / "progress.jsonl")
    journal.start(FP, resume=False)
    journal.append(record)
    before = journal.path.read_bytes()
    with pytest.raises(ShardError, match="record 1 is not a completed task"):
        journal.start(FP, resume=True)
    assert journal.path.read_bytes() == before
