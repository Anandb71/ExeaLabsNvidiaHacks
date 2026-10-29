"""scans/<room>/scan.json is loaded under its folder's name, as `demo`, `summary` and the upload worker name it.
A room started as a copy of another room's folder still says that room's room_id inside, and `arrival sim --room
my_kitchen` wrote out/datasets/hero_kitchen_sim (and its report and journal) over the real hero kitchen's."""

import argparse
import json
import shutil
from pathlib import Path

import pytest

from arrival.scan import bundle

REPO = Path(__file__).resolve().parents[1]
HERO = REPO / "scans" / "hero_kitchen" / "scan.json"


def _copied(tmp_path) -> Path:
    room = tmp_path / "scans" / "my_kitchen"
    room.mkdir(parents=True)
    shutil.copy(HERO, room / "scan.json")  # room_id still "hero_kitchen"
    return tmp_path / "scans"


def test_a_copied_scan_takes_its_folder_name(tmp_path):
    b = bundle.load_room("my_kitchen", _copied(tmp_path))
    assert b["room_id"] == "my_kitchen" and b["source_room_id"] == "hero_kitchen"


def test_sim_and_demo_name_the_twin_after_the_folder(tmp_path):
    pytest.importorskip("mujoco")
    from arrival import cli

    scans, out = _copied(tmp_path), tmp_path / "out"
    ns = argparse.Namespace(
        room="my_kitchen",
        scans=str(scans),
        out=str(out),
        families="wipe_counter",
        episodes=1,
        people=0,
        res=64,
        seed=0,
        base="fixed",
        arm="procedural",
    )
    _, twin = cli._sim_args(ns)
    assert twin["room_id"] == "my_kitchen"  # collect writes out/datasets/<room_id>_sim
    cli.run_demo("my_kitchen", 1, "none", out, scans, log=lambda *_: None)
    assert json.loads((out / "twins" / "my_kitchen" / "twin.json").read_text())["room_id"] == "my_kitchen"


def test_a_scan_in_its_own_folder_is_unchanged():
    assert bundle.load_room("hero_kitchen", REPO / "scans") == bundle.load(HERO)
