"""L15: twin.json and the dataset meta say what they came from (input scan hash, code sha, seeds, generators), and
objects the pipeline invented are labelled so, never shown as measured."""

import hashlib
import json
import subprocess

import pytest
from conftest import REPO, SCANS

from arrival import provenance
from arrival.scan import bundle as scan_bundle
from arrival.scan import roomplan
from arrival.twin.compile import compile_twin, place_default_objects, write_twin

HERO = SCANS / "hero_kitchen"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_twin_records_the_input_file_hash_and_a_bundle_hash():
    twin = compile_twin(scan_bundle.load(HERO / "scan.json"))
    p = twin["provenance"]
    assert p["schema_version"] == 1
    assert p["input"]["file_sha256"] == _sha(HERO / "scan.json")
    assert len(p["input"]["bundle_sha256"]) == 64
    assert (
        p["generators"]["geometry"] == "authored_sample" and p["generators"]["twin_compiler"] == "arrival.twin.compile"
    )


def test_changing_one_byte_of_the_scan_changes_both_hashes(tmp_path):
    raw = json.loads((HERO / "scan.json").read_text())
    a = tmp_path / "a.json"
    a.write_text(json.dumps(raw))
    raw["fixtures"][0]["position"][0] += 0.001
    b = tmp_path / "b.json"
    b.write_text(json.dumps(raw))
    pa, pb = (compile_twin(scan_bundle.load(p))["provenance"]["input"] for p in (a, b))
    assert pa["file_sha256"] != pb["file_sha256"] and pa["bundle_sha256"] != pb["bundle_sha256"]
    assert pa == compile_twin(scan_bundle.load(a))["provenance"]["input"]  # and the same file hashes the same


def test_roomplan_load_hashes_the_export_as_received():
    twin = compile_twin(roomplan.load(HERO / "roomplan.json", "hero_kitchen"))
    assert twin["provenance"]["input"]["file_sha256"] == _sha(HERO / "roomplan.json")
    assert twin["provenance"]["generators"]["geometry"] == "roomplan"


def test_bundle_hash_ignores_the_input_hash_field():
    b = scan_bundle.load(HERO / "scan.json")
    other = dict(b, input_sha256="0" * 64)
    assert provenance.input_record(b)["bundle_sha256"] == provenance.input_record(other)["bundle_sha256"]


def test_git_sha_matches_the_checkout():
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no git checkout")
    state = provenance.git_state()
    assert state["sha"] == head and state["source"] == "git" and isinstance(state["dirty"], bool)


def test_git_sha_env_override_and_missing_git(monkeypatch, tmp_path):
    monkeypatch.setenv("ARRIVAL_GIT_SHA", "abc1234")
    assert provenance.git_state() == {"sha": "abc1234", "dirty": None, "source": "env"}
    monkeypatch.delenv("ARRIVAL_GIT_SHA")
    nowhere = provenance.git_state(tmp_path)  # not a repository: unknown, never a made-up value
    assert nowhere == {"sha": None, "dirty": None, "source": None}


def test_default_objects_are_labelled_synthetic_with_their_seed():
    twin = compile_twin(roomplan.load(HERO / "roomplan.json", "hero_kitchen"))
    assert twin["objects"] == []
    placed = place_default_objects(twin, seed=7)
    assert placed and {o["provenance"] for o in placed} == {"default_synthetic"}
    assert twin["provenance"]["seeds"] == {"default_objects": 7}
    assert twin["provenance"]["generators"]["objects"] == {
        "detected": 0,
        "authored": 0,
        "default_synthetic": len(placed),
    }


def test_authored_objects_stay_authored_and_nothing_is_called_detected():
    twin = compile_twin(scan_bundle.load(HERO / "scan.json"))
    assert twin["objects"] and {o["provenance"] for o in twin["objects"]} == {"authored"}
    assert twin["provenance"]["generators"]["objects"]["detected"] == 0


def test_a_recompile_keeps_the_synthetic_label():
    first = compile_twin(roomplan.load(HERO / "roomplan.json", "hero_kitchen"))
    b = roomplan.load(HERO / "roomplan.json", "hero_kitchen")
    b["objects"] = place_default_objects(first, seed=0)  # what `arrival demo --from-roomplan` does
    again = compile_twin(b)
    assert {o["provenance"] for o in again["objects"]} == {"default_synthetic"}
    assert again["provenance"]["generators"]["objects"]["default_synthetic"] == len(again["objects"])


def test_twin_json_round_trips_provenance(tmp_path):
    twin = compile_twin(scan_bundle.load(HERO / "scan.json"))
    path, _ = write_twin(twin, tmp_path)
    assert json.loads(path.read_text())["provenance"] == twin["provenance"]


def test_place_default_objects_on_an_old_twin_without_provenance():
    twin = compile_twin(roomplan.load(HERO / "roomplan.json", "hero_kitchen"))
    del twin["provenance"]
    assert place_default_objects(twin, seed=1)
    assert twin["provenance"]["seeds"] == {"default_objects": 1}


def test_run_record_combines_twin_and_run():
    twin = compile_twin(scan_bundle.load(HERO / "scan.json"))
    rec = provenance.run_record(twin, seed=5, generator="mujoco_scripted_ik", policy="scripted", episodes=3)
    assert rec["input"]["file_sha256"] == _sha(HERO / "scan.json")
    assert rec["run"] == {"seed": 5, "generator": "mujoco_scripted_ik", "policy": "scripted", "episodes": 3}
    assert rec["run_code"] == provenance.git_state() and rec["twin_generators"]["geometry"] == "authored_sample"
    json.dumps(rec)  # plain JSON


def test_sim_dataset_meta_carries_the_record(tmp_path):
    pytest.importorskip("mujoco")
    from arrival.export.validate import validate
    from arrival.sim.collect import collect

    twin = compile_twin(scan_bundle.load(HERO / "scan.json"))
    res = collect(twin, ["open_drawer"], 1, tmp_path, people=0, res=32, seed=11, log=lambda *_: None)
    info = json.loads((res["dataset"] / "meta" / "info.json").read_text())
    prov = info["arrival_provenance"]
    assert prov["input"]["file_sha256"] == _sha(HERO / "scan.json")
    assert prov["run"]["seed"] == 11 and prov["run"]["generator"] == info["arrival_generator"] == "mujoco_scripted_ik"
    assert prov["run"]["families"] == ["open_drawer"] and prov["run_code"]["sha"] == provenance.git_state()["sha"]
    assert validate(res["dataset"]) == []


def test_scripted_demo_dataset_meta_carries_the_record(tmp_path):
    from arrival.cli import run_demo
    from arrival.export.validate import validate

    res = run_demo("hero_kitchen", 1, "none", tmp_path, SCANS, from_roomplan=True, seed=2, log=lambda *_: None)
    info = json.loads((res["dataset"] / "meta" / "info.json").read_text())
    prov = info["arrival_provenance"]
    assert prov["input"]["file_sha256"] == _sha(HERO / "roomplan.json")
    assert prov["twin_seeds"] == {"default_objects": 2} and prov["run"]["generator"] == "scripted_retarget"
    assert info["arrival_physically_valid"] is False  # provenance does not change what the data claims
    twin = json.loads((res["twin_dir"] / "twin.json").read_text())
    assert twin["provenance"]["generators"]["objects"]["default_synthetic"] == len(twin["objects"])
    assert validate(res["dataset"]) == []
