"""A number written as a string in an upload is a bad scan, not a worker fault.

`bundle.numbers` used float(), which also reads "2.4". A scan.json wall start of
["0", 0.0] then passed the field check and raised TypeError in the wall-span check,
and the worker queued the same upload again (the default is two retries). A wall
height of "2.4" passed and the twin kept the str, so the sim failed on it later.
"""

import json

import pytest
from conftest import SCANS

from arrival.jobs.limits import Limits
from arrival.jobs.queue import create_job
from arrival.jobs.worker import load_upload, retryable, run_once
from arrival.scan.bundle import ScanBundleError

EYE = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
QUIET = lambda *_: None  # noqa: E731


def _scan(mutate) -> bytes:
    data = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text(encoding="utf-8"))
    mutate(data)
    return json.dumps(data).encode()


def _roomplan(mutate) -> bytes:
    data = {
        "walls": [{"identifier": "w", "dimensions": [4.0, 2.5, 0.1], "transform": list(EYE)}],
        "objects": [
            {"identifier": "table", "category": "table", "dimensions": [1.0, 1.0, 1.0], "transform": list(EYE)}
        ],
    }
    mutate(data)
    return json.dumps(data).encode()


REFUSED = {
    "scan wall start": (lambda: _scan(lambda d: d["walls"][0]["start"].__setitem__(0, "0")), "wall wall_north"),
    "scan wall height": (lambda: _scan(lambda d: d["walls"][0].__setitem__("height", "2.4")), "wall wall_north"),
    "scan object position": (
        lambda: _scan(lambda d: d["objects"][0]["position"].__setitem__(0, "1.0")),
        "object 'mug_1': position",
    ),
    "scan object yaw": (lambda: _scan(lambda d: d["objects"][0].__setitem__("yaw", "0.5")), "object 'mug_1': yaw"),
    "scan fixture size": (
        lambda: _scan(lambda d: d["fixtures"][0]["size"].__setitem__(0, "1.45")),
        "fixture 'counter_a': size",
    ),
    "roomplan transform": (lambda: _roomplan(lambda d: d["walls"][0]["transform"].__setitem__(12, "0.5")), "transform"),
}


@pytest.mark.parametrize("build,field", REFUSED.values(), ids=REFUSED.keys())
def test_a_number_written_as_a_string_is_a_scan_error_and_is_not_retried(build, field, tmp_path):
    upload = tmp_path / "upload.bin"
    upload.write_bytes(build())
    with pytest.raises(ScanBundleError) as caught:
        load_upload(upload, "room_job")
    assert field in str(caught.value)
    assert not retryable(caught.value)


def test_worker_does_not_requeue_a_scan_with_a_string_wall_start(tmp_path):
    out = tmp_path / "out"
    create_job(out / "jobs", "k.json", data=REFUSED["scan wall start"][0]())
    status = run_once(out, limits=Limits(job_retry_backoff=0), log=QUIET)
    assert status["status"] == "failed" and status["job_attempts"] == 1 and "retry_errors" not in status
    assert status["error"].startswith("ScanBundleError: wall wall_north: start and end must be [x, y] numbers")
    assert run_once(out, limits=Limits(job_retry_backoff=0), log=QUIET) is None


def test_plain_numbers_still_load():
    """The fixture scan and an int coordinate are unchanged: only a str (or bool) is refused."""
    bundle = json.loads(_scan(lambda d: d["objects"][0]["position"].__setitem__(0, 1)))
    from arrival.scan.bundle import validate

    assert validate(bundle)["objects"][0]["position"][0] == 1
