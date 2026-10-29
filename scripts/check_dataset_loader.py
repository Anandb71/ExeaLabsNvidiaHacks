"""Offline GR00T/LeRobot loader conformance check (no gr00t import). Exits 0 if the dataset passes, 1 if not.

.venv/bin/python scripts/check_dataset_loader.py out/datasets/hero_kitchen_sim [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrival.export.loader_check import check_loader, format_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset_dir")
    p.add_argument("--json", action="store_true", help="print the full result as JSON instead of the short report")
    a = p.parse_args(argv)
    res = check_loader(a.dataset_dir)
    print(json.dumps(res, indent=2) if a.json else format_report(a.dataset_dir, res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
