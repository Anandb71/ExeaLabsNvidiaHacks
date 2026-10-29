"""A scan.json wall needs an id of its own: a non-empty string that no other wall and no object has. The sim's MuJoCo
model names each wall's box and each object's geom after its id, so a repeated id failed the model build of every
task, which the upload worker reported as an unsupported layout. A wall with no id passed validation and the job then
failed with KeyError: 'id' when the twin was written, which the worker retried as a machine fault."""

import json

import pytest
from conftest import SCANS

from arrival.jobs.worker import load_upload, retryable
from arrival.scan import bundle as scan_bundle

BAD = {
    "two walls with one id": lambda d: d["walls"][1].update(id=d["walls"][0]["id"]),
    "a wall named like an object": lambda d: d["walls"][0].update(id=d["objects"][0]["id"]),
    "a wall without an id": lambda d: d["walls"][0].pop("id"),
    "an empty wall id": lambda d: d["walls"][0].update(id=""),
    "a wall id that is a number": lambda d: d["walls"][0].update(id=7),
}


def _hero(mutate=None):
    data = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(data)
    return data


@pytest.mark.parametrize("mutate", BAD.values(), ids=BAD.keys())
def test_bad_wall_id_is_a_scan_bundle_error(mutate):
    with pytest.raises(scan_bundle.ScanBundleError, match="wall"):
        scan_bundle.validate(_hero(mutate))


@pytest.mark.parametrize("mutate", BAD.values(), ids=BAD.keys())
def test_upload_with_a_bad_wall_id_is_refused_and_not_retried(mutate, tmp_path):
    upload = tmp_path / "upload.bin"
    upload.write_text(json.dumps(_hero(mutate)), encoding="utf-8")
    with pytest.raises(scan_bundle.ScanBundleError) as e:
        load_upload(upload, "room_job")
    assert not retryable(e.value)


def test_a_wall_may_share_an_id_with_a_fixture():
    """A fixture is a MuJoCo body and a wall a geom, so those names do not clash."""
    data = _hero(lambda d: d["walls"][0].update(id=d["fixtures"][0]["id"]))
    assert scan_bundle.validate(data)["walls"][0]["id"] == "counter_a"
    assert [w["id"] for w in scan_bundle.validate(_hero())["walls"]] == [
        "wall_north",
        "wall_east",
        "wall_south",
        "wall_west",
    ]
