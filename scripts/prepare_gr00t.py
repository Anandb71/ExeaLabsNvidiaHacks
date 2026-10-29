"""Check an Arrival sim dataset and print a GR00T N1.7 fine-tune command.

This prepares a run; it never downloads weights, starts training, or reports a
loader result. Run the printed command in an Isaac-GR00T environment with a GPU.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrival.export.validate import validate

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "gr00t"
VIDEO = {"ego_view", "wrist"}
ARM = {"single_arm", "gripper"}


def prepare(dataset: str | Path, gr00t_root: str | Path, output: str | Path, steps: int = 100) -> list[str]:
    dataset, gr00t_root, output = Path(dataset).resolve(), Path(gr00t_root).resolve(), Path(output).resolve()
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not (gr00t_root / "gr00t" / "experiment" / "launch_finetune.py").is_file():
        raise ValueError(f"{gr00t_root} is not an Isaac-GR00T checkout with launch_finetune.py")
    problems = validate(dataset)
    if problems:
        raise ValueError("dataset validation failed:\n" + "\n".join(problems[:10]))
    info = json.loads((dataset / "meta" / "info.json").read_text())
    embodiment = json.loads((dataset / "meta" / "embodiment.json").read_text())
    modality = json.loads((dataset / "meta" / "modality.json").read_text())
    if info.get("arrival_physically_valid") is not True or embodiment.get("joints_are_placeholder") is not False:
        raise ValueError("fine-tuning requires physically valid episodes with real simulated joint states")
    if info.get("arrival_video_policy") != "all" or info.get("total_episodes", 0) < 1:
        raise ValueError("fine-tuning requires at least one episode with video for every episode")
    mobile = embodiment.get("base_actuated") is True
    keys = ARM | ({"base"} if mobile else set())
    config = CONFIG_DIR / ("arrival_panda_mobile_config.py" if mobile else "arrival_panda_config.py")
    if not config.is_file():
        raise ValueError(f"missing GR00T modality config {config}")
    if not VIDEO <= set(modality.get("video", {})):
        raise ValueError(f"modality.json video keys must include {sorted(VIDEO)}")
    for group in ("state", "action"):
        if not keys <= set(modality.get(group, {})):
            raise ValueError(f"modality.json {group} keys must include {sorted(keys)}")
    if "human.task_description" not in modality.get("annotation", {}):
        raise ValueError("modality.json lacks the human.task_description language annotation")
    if embodiment.get("control") != "absolute_joint_position" or any(
        modality["action"][key].get("absolute") is not True for key in keys
    ):
        raise ValueError("GR00T config expects absolute joint, gripper and base targets in the dataset")
    return [
        "python",
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        "nvidia/GR00T-N1.7-3B",
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(config),
        "--num-gpus",
        "1",
        "--output-dir",
        str(output),
        "--max-steps",
        str(steps),
        "--global-batch-size",
        "8",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--gr00t-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("out/checkpoints/gr00t"))
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        command = prepare(args.dataset, args.gr00t_root, args.output, args.steps)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"GR00T preflight: {exc}\n")
    print(f"cd {shlex.quote(str(args.gr00t_root.resolve()))}")
    print(shlex.join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
