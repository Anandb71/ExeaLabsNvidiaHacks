"""The fine-tune command must match current sim data and reject placeholder data."""

import json
from pathlib import Path

import pytest

from scripts import prepare_gr00t


def _dataset(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"arrival_physically_valid": True, "arrival_video_policy": "all", "total_episodes": 2})
    )
    (meta / "embodiment.json").write_text(
        json.dumps({"joints_are_placeholder": False, "base_actuated": False, "control": "absolute_joint_position"})
    )
    keys = {k: {"absolute": True} for k in ("single_arm", "gripper", "base")}
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "video": {k: {} for k in ("ego_view", "wrist")},
                "state": keys,
                "action": keys,
                "annotation": {"human.task_description": {}},
            }
        )
    )
    monkeypatch.setattr(prepare_gr00t, "validate", lambda _: [])
    checkout = tmp_path / "Isaac-GR00T" / "gr00t" / "experiment"
    checkout.mkdir(parents=True)
    (checkout / "launch_finetune.py").touch()
    return root, checkout.parents[1]


def test_prepare_matches_dataset_modalities_and_stops_placeholder_runs(tmp_path, monkeypatch):
    root, checkout = _dataset(tmp_path, monkeypatch)
    command = prepare_gr00t.prepare(root, checkout, tmp_path / "weights", steps=100)
    assert command[command.index("--dataset-path") + 1] == str(root)
    assert (
        Path(command[command.index("--modality-config-path") + 1])
        .as_posix()
        .endswith("configs/gr00t/arrival_panda_config.py")
    )
    assert command[command.index("--max-steps") + 1] == "100"
    path = root / "meta" / "embodiment.json"
    path.write_text(json.dumps({"joints_are_placeholder": True, "control": "absolute_joint_position"}))
    with pytest.raises(ValueError, match="physically valid"):
        prepare_gr00t.prepare(root, checkout, tmp_path / "weights")


def test_prepare_refuses_wrong_modality_and_sparse_video(tmp_path, monkeypatch):
    root, checkout = _dataset(tmp_path, monkeypatch)
    info = root / "meta" / "info.json"
    info.write_text(
        json.dumps({"arrival_physically_valid": True, "arrival_video_policy": "sparse", "total_episodes": 2})
    )
    with pytest.raises(ValueError, match="video for every episode"):
        prepare_gr00t.prepare(root, checkout, tmp_path / "weights")
    info.write_text(json.dumps({"arrival_physically_valid": True, "arrival_video_policy": "all", "total_episodes": 2}))
    modality = root / "meta" / "modality.json"
    raw = json.loads(modality.read_text())
    del raw["state"]["single_arm"]
    modality.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="state keys"):
        prepare_gr00t.prepare(root, checkout, tmp_path / "weights")


def test_prepare_selects_existing_mobile_config(tmp_path, monkeypatch):
    root, checkout = _dataset(tmp_path, monkeypatch)
    path = root / "meta" / "embodiment.json"
    path.write_text(
        json.dumps({"joints_are_placeholder": False, "base_actuated": True, "control": "absolute_joint_position"})
    )
    command = prepare_gr00t.prepare(root, checkout, tmp_path / "weights")
    assert (
        Path(command[command.index("--modality-config-path") + 1])
        .as_posix()
        .endswith("configs/gr00t/arrival_panda_mobile_config.py")
    )
    modality = root / "meta" / "modality.json"
    raw = json.loads(modality.read_text())
    del raw["action"]["base"]
    modality.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="action keys"):
        prepare_gr00t.prepare(root, checkout, tmp_path / "weights")
