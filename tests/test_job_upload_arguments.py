import io

import pytest

from arrival.jobs.queue import create_job


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"data": b"x", "src": io.BytesIO(b"y")},
        {"src": io.BytesIO(b"x"), "offset": -1},
        {"src": io.BytesIO(b"x"), "length": -1},
        {"data": b"x", "length": True},
    ],
)
def test_bad_upload_arguments_leave_no_job(kwargs, tmp_path):
    jobs = tmp_path / "jobs"
    with pytest.raises(ValueError):
        create_job(jobs, "scan.json", **kwargs)
    assert not jobs.exists()
