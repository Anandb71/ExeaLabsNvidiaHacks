import json

import pytest

from arrival.jobs.fanout import FanoutConfig
from arrival.jobs.fanout_status import StatusRecorder
from arrival.jobs.shards import ShardError

CFG = FanoutConfig("run", 2, "room", "counter_to_sink", 4, 0, 32, 0, "scans")


class Store:
    def __init__(self, value):
        self.value = value

    def exists(self, key):
        return True

    def get_object(self, key):
        return json.dumps(self.value).encode()


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"run_id": "other", "shard": 0, "shards": 2, "attempts": []},
        {"run_id": "run", "shard": 0, "shards": 2, "attempts": [None]},
    ],
)
def test_status_recorder_rejects_malformed_saved_record(value):
    with pytest.raises(ShardError, match="invalid status record"):
        StatusRecorder(Store(value))._record(CFG, 0)
