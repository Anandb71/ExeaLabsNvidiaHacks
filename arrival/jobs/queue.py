"""Local job queue: store an upload and write a queued status record.

``arrival.jobs.worker`` (``python -m arrival work``) drains it. The status JSON is
the whole API. This module must not import the compiler or generator.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import BinaryIO

from arrival.jobs.limits import MAX_UPLOAD_BYTES, QueueFull

CHUNK = 1 << 20


def check_upload_size(content_length: int, limit: int = MAX_UPLOAD_BYTES) -> bool:
    """Return True if an upload of ``content_length`` bytes is accepted."""
    return 0 <= int(content_length) <= int(limit)


def new_job_id() -> str:
    """``job_`` + 128 random bits, so an id is a capability: it cannot be guessed or enumerated."""
    return "job_" + secrets.token_hex(16)


def count_active(jobs_root: Path) -> int:
    """Jobs that are ``queued`` or ``running`` (unreadable status files are not counted)."""
    jobs_root = Path(jobs_root)
    n = 0
    for d in jobs_root.iterdir() if jobs_root.is_dir() else []:
        try:
            st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        n += isinstance(st, dict) and st.get("status") in ("queued", "running")
    return n


def create_job(
    jobs_root: Path,
    filename: str,
    data: bytes | None = None,
    src: BinaryIO | None = None,
    offset: int = 0,
    length: int = 0,
    max_queue: int = 0,
    extra_status: dict | None = None,
    adopt: dict[str, Path] | None = None,
) -> dict:
    """Write ``<jobs_root>/<id>/upload.bin`` (from ``data``, or ``length`` bytes of ``src`` at ``offset``,
    copied in chunks) and ``status.json``; return the status. ``adopt`` maps a name to a prepared folder that is
    moved to ``<id>/<name>`` before the status exists, so a worker never sees a queued job without it;
    ``extra_status`` keys are merged into the status. With ``max_queue`` > 0, raise ``QueueFull`` when that
    many jobs are already queued or running. The count is not atomic with the write, so concurrent uploads can
    overshoot the limit by a few; it bounds the queue, it is not an exact gate."""
    if (data is None) == (src is None):
        raise ValueError("provide exactly one of data or src")
    if type(offset) is not int or offset < 0 or type(length) is not int or length < 0:
        raise ValueError("offset and length must be nonnegative whole numbers")
    jobs_root = Path(jobs_root)
    if max_queue and count_active(jobs_root) >= max_queue:
        raise QueueFull(f"queue is full: {max_queue} jobs are queued or running")
    job_id = new_job_id()
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    if data is not None:
        (job_dir / "upload.bin").write_bytes(data)
        length = len(data)
    else:
        src.seek(offset)
        with open(job_dir / "upload.bin", "wb") as out:
            left = length
            while left > 0:
                chunk = src.read(min(left, CHUNK))
                if not chunk:
                    break
                out.write(chunk)
                left -= len(chunk)
        if left:
            shutil.rmtree(job_dir)
            raise EOFError(f"upload ended {left} bytes before its declared length")
    for name, folder in (adopt or {}).items():
        shutil.move(str(folder), str(job_dir / name))
    status = {
        **(extra_status or {}),
        "id": job_id,
        "status": "queued",
        "created_unix": int(time.time()),
        "filename": Path(filename or "upload.bin").name,
        "bytes": length,
    }
    return write_status(job_dir, status)


def write_status(job_dir: Path, status: dict) -> dict:
    """Replace ``<job_dir>/status.json`` in one step, so the web page and the capture app never read a half-written
    file while the worker updates it. On Windows the replace fails while a reader has the file open; retry briefly."""
    path = Path(job_dir) / "status.json"
    tmp = path.with_name(f".status.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    for _ in range(100):
        try:
            os.replace(tmp, path)
            return status
        except PermissionError:
            time.sleep(0.005)
    os.replace(tmp, path)
    return status


def read_status(job_dir: Path) -> dict | None:
    """The job's status.json as a dict, or None when it is missing, unreadable or not a JSON object."""
    try:
        st = json.loads((Path(job_dir) / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return st if isinstance(st, dict) else None


def read_job(jobs_root: Path, job_id: str) -> dict | None:
    """A job's status.json, or None for an id that is not a job id (so no path outside jobs/ is ever read) or a job that
    does not exist. Retries briefly while Windows has the file locked mid-replace."""
    if not isinstance(job_id, str) or re.fullmatch(r"job_[0-9a-f]{32}", job_id) is None:
        return None
    path = Path(jobs_root) / job_id / "status.json"
    for _ in range(100):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:  # Windows: write_status is swapping the file in right now
            time.sleep(0.005)
    return json.loads(path.read_text(encoding="utf-8"))
