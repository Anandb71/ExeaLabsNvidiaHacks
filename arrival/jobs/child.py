"""Child process for one claimed job: ``python -m arrival.jobs.child JOB_DIR OUT_DIR OPTIONS_JSON``.

``arrival.jobs.worker.run_isolated`` starts this so the worker can kill a job that exceeds its wall-clock or memory limit.
The job's own failure is recorded in ``status.json`` and does not change the exit code; a non-zero exit means the
process itself died, which the worker records as ``WorkerProcessError``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from arrival.jobs.limits import Limits
from arrival.jobs.queue import read_status
from arrival.jobs.worker import process


def main(argv: list[str] | None = None) -> int:
    job_dir, out_dir, raw = (argv if argv is not None else sys.argv[1:])[:3]
    opts = json.loads(raw)
    limits = Limits(**opts.pop("limits", {}))
    status = read_status(Path(job_dir))
    if status is None:
        print(f"[work] {job_dir}: no readable status.json", file=sys.stderr)
        return 3
    process(Path(job_dir), status, Path(out_dir), limits=limits, **opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
