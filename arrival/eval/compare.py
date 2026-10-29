"""Paired comparison of two `eval` runs that replayed the same manifest (docs/07-eval.md, "Replaying identical trials").

Trials are paired by `trial_index`. Only trials where exactly one run succeeded carry information: the exact
two-sided sign test is over those. No dependency on the simulator, so it runs on records copied off a GPU box.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path


class CompareError(ValueError):
    pass


def load_rows(run: str | Path) -> list[dict]:
    """The trial rows of one eval run (its folder or its episodes.jsonl), checked to be an eval run's rows (not a
    LeRobot dataset's file of the same name) with unique trial indexes. Raises CompareError otherwise."""
    path = Path(run)
    if path.is_dir():
        path = path / "episodes.jsonl"
    if not path.is_file():
        raise CompareError(f"no episodes.jsonl at {run}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as e:
        raise CompareError(f"cannot read {path}: {e}") from None
    if not rows:
        raise CompareError(f"{path} has no trials")
    need = ("manifest_id", "checkpoint_id", "trial_index", "task_id", "family", "success")
    seen_trials = set()
    for i, r in enumerate(rows):
        missing = [k for k in need if not isinstance(r, dict) or k not in r]
        if missing:  # a LeRobot dataset's meta/episodes.jsonl has the same name
            raise CompareError(f"{path} is not an eval run's episodes.jsonl (row {i} has no {', '.join(missing)})")
        if type(r["success"]) is not bool:
            raise CompareError(f"{path}: row {i} success must be a bool")
        index = r["trial_index"]
        if type(index) is not int or index < 0:
            raise CompareError(f"{path}: row {i} trial_index must be a nonnegative integer")
        if index in seen_trials:
            raise CompareError(f"{path}: duplicate trial_index {index}")
        seen_trials.add(index)
    if len({r.get("manifest_id") for r in rows}) != 1:
        raise CompareError(f"{path} mixes manifests")
    if any(row["checkpoint_id"] != rows[0]["checkpoint_id"] for row in rows):
        raise CompareError(f"{path} mixes checkpoints")
    return sorted(rows, key=lambda r: r["trial_index"])


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial p-value of wins:losses under a fair coin; 1.0 when there are no discordant trials."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def compare(run_a: str | Path, run_b: str | Path) -> dict:
    """Pair the trials of two runs that replayed the same manifest and count, per family and in total, the successes of
    each and the trials only one of them won, with the exact sign test's p. Raises CompareError when the trials do
    not pair up."""
    a, b = load_rows(run_a), load_rows(run_b)
    if a[0]["manifest_id"] != b[0]["manifest_id"]:
        raise CompareError("the runs replayed different manifests; their trials are not the same")
    if [r["trial_index"] for r in a] != [r["trial_index"] for r in b] or [r["task_id"] for r in a] != [
        r["task_id"] for r in b
    ]:
        raise CompareError("the runs did not score the same trials (one is incomplete or reordered)")

    if any(x["family"] != y["family"] for x, y in zip(a, b)):
        raise CompareError("the runs disagree on task families for paired trials")

    def block(pairs):
        wins = sum(x["success"] and not y["success"] for x, y in pairs)
        losses = sum(y["success"] and not x["success"] for x, y in pairs)
        n = len(pairs)
        ka, kb = sum(x["success"] for x, _ in pairs), sum(y["success"] for _, y in pairs)
        return {
            "trials": n,
            "a_successes": ka,
            "b_successes": kb,
            "a_rate": round(ka / n, 4),
            "b_rate": round(kb / n, 4),
            "b_won": losses,
            "a_won": wins,
            "p": sign_test_p(wins, losses),
        }

    pairs = list(zip(a, b))
    families = {}
    for fam in sorted({x["family"] for x, _ in pairs}):
        families[fam] = block([(x, y) for x, y in pairs if x["family"] == fam])
    return {
        "manifest_id": a[0]["manifest_id"],
        "a_checkpoint": a[0]["checkpoint_id"],
        "b_checkpoint": b[0]["checkpoint_id"],
        "total": block(pairs),
        "families": families,
    }


def table(res: dict) -> str:
    """The comparison as a fixed-width text table, one row per family and a total row."""
    head = f"A = {res['a_checkpoint']}   B = {res['b_checkpoint']}   manifest {res['manifest_id'][:12]}"
    lines = [head, f"{'':<20}{'trials':>7}{'A':>7}{'B':>7}{'A won':>7}{'B won':>7}{'p':>9}"]
    for name, r in {**res["families"], "total": res["total"]}.items():
        lines.append(
            f"{name:<20}{r['trials']:>7}{r['a_successes']:>7}{r['b_successes']:>7}{r['a_won']:>7}{r['b_won']:>7}"
            f"{r['p']:>9.3f}"
        )
    lines.append("p: exact two-sided sign test on the trials where exactly one run succeeded")
    return "\n".join(lines)
