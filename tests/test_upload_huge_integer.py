"""An integer too large for float() in an upload is a bad scan, not a worker fault.

`bundle.numbers` and the RoomPlan checks call `float` or `math.isfinite` on JSON
numbers. `10**400` raises OverflowError, which is not a ValueError, so the worker
queues the same upload again (the default is two retries). NaN and a string in the
same field are already ScanBundleError and are not retried.
"""

import json

import pytest
from conftest import SCANS

from arrival.jobs.limits import Limits
from arrival.jobs.queue import create_job
from arrival.jobs.worker import load_upload, retryable, run_once
from arrival.scan.bundle import ScanBundleError
from arrival.scan.roomplan import to_bundle

HUGE = 10**400
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
    "scan object position": lambda: _scan(lambda d: d["objects"][0]["position"].__setitem__(0, HUGE)),
    "scan yaw": lambda: _scan(lambda d: d["objects"][0].__setitem__("yaw", HUGE)),
    "scan wall start": lambda: _scan(lambda d: d["walls"][0]["start"].__setitem__(0, HUGE)),
    "roomplan transform": lambda: _roomplan(lambda d: d["walls"][0]["transform"].__setitem__(12, HUGE)),
    "roomplan wall dimensions": lambda: _roomplan(lambda d: d["walls"][0]["dimensions"].__setitem__(0, HUGE)),
    "roomplan object dimensions": lambda: _roomplan(lambda d: d["objects"][0]["dimensions"].__setitem__(0, HUGE)),
    "roomplan door dimensions": lambda: _roomplan(
        lambda d: d.__setitem__("doors", [{"identifier": "d", "dimensions": [HUGE, 2.0, 0.1], "transform": list(EYE)}])
    ),
    "roomplan floor polygon": lambda: _roomplan(
        lambda d: d.__setitem__("floors", [{"identifier": "f", "polygonCorners": [[0, 0, 0], [1, 0, 0], [HUGE, 1, 0]]}])
    ),
}


@pytest.mark.parametrize("build", REFUSED.values(), ids=REFUSED.keys())
def test_huge_integer_is_a_scan_error_and_is_not_retried(build, tmp_path):
    upload = tmp_path / "upload.bin"
    upload.write_bytes(build())
    with pytest.raises(ScanBundleError) as caught:
        load_upload(upload, "room_job")
    assert not retryable(caught.value)


def test_worker_does_not_requeue_a_scan_with_a_huge_integer(tmp_path):
    out = tmp_path / "out"
    create_job(out / "jobs", "k.json", data=REFUSED["scan object position"]())
    status = run_once(out, limits=Limits(job_retry_backoff=0), log=QUIET)
    assert status["status"] == "failed" and status["job_attempts"] == 1 and "retry_errors" not in status
    assert status["error"].startswith("ScanBundleError:")
    assert run_once(out, limits=Limits(job_retry_backoff=0), log=QUIET) is None


@pytest.mark.parametrize(
    "curve",
    [
        {"startAngle": HUGE, "endAngle": 1.0, "radius": 2.0},
        {"startAngle": 0.0, "endAngle": 1.0, "radius": HUGE},
        {"startAngle": 0.0, "endAngle": 1.0, "radius": 2.0, "center": [HUGE, 0.0]},
    ],
)
def test_a_huge_integer_in_a_curve_is_treated_as_not_finite(curve):
    """A non-finite curve angle or radius is dropped and the wall stays straight; a non-finite centre is ignored.
    A huge int used to raise OverflowError on the way there."""
    wall = {"identifier": "w", "dimensions": [4.0, 2.5, 0.1], "transform": list(EYE)}
    plain = {"walls": [dict(wall)], "objects": []}
    curved = {"walls": [{**wall, "curve": curve}], "objects": []}
    if "center" in curve:
        plain["walls"][0]["curve"] = {"startAngle": 0.0, "endAngle": 1.0, "radius": 2.0}

    def outcome(data):
        try:
            return to_bundle(data, "r")
        except ScanBundleError as e:
            return str(e)

    assert outcome(curved) == outcome(plain)
