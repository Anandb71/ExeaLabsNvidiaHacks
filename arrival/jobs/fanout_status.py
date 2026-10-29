"""Per-shard status records for a fan-out run (C1, issue #66).

`StatusRecorder(backend)` is itself a `Backend`: it passes every call through to `backend` and keeps one record per
shard at `<run>/status/<i>of<n>.json`, in the backend's own storage next to `<run>/run.json`, so a record lives in
the bucket for NebiusBackend and in the store directory for LocalBackend. The record is rewritten once an attempt has
a job id (or has failed to get one), at each state change and when it ends, so a kill loses at most the last change. Per attempt it keeps the job id, the state
history, `submitted_at` and `finished_at` (unix seconds), `wall_seconds` (the orchestrator's clock from submit until
the marker is written or the failure is seen, so it includes provisioning and the copy to the shard prefix), the
resources the backend asked for (`None` where there are none, as for a local subprocess), the outcome, and for a
failed attempt `failed_stage` (submit, run, fetch or marker) with the reason. `fetch` is also a succeeded job whose
copied files have no `meta/info.json` or `meta/episodes.jsonl`: that check happens before the attempt is noted as
ready for its done marker, so the reason is the missing path. `marker` is the done-marker write, or the fan-out
stopping after those files are present. A rerun loads the record and appends;
an attempt an earlier process left open becomes `abandoned`. Reasons pass through `config.redact`. Nothing is priced.
"""

from __future__ import annotations

import json
import time

from arrival import config as settings
from arrival.jobs.fanout import FAILED, RUNNING, SUCCEEDED, FanoutConfig, marker_key, require_shard_artifacts
from arrival.jobs.shards import ShardError


def status_key(cfg: FanoutConfig, i: int) -> str:
    return f"{cfg.run_id}/status/{i}of{cfg.shards}.json"


class StatusRecorder:
    """A Backend that passes every call to `backend` and keeps a per-shard status record (attempts, states, timing,
    outcome, failure stage) in the backend's own storage; see the module docstring for the record's fields."""

    def __init__(self, backend):
        self.backend = backend
        self._records: dict[str, dict] = {}  # status key -> record
        self._open: dict[str, tuple] = {}  # job id -> (cfg, shard, attempt entry, monotonic start)
        self._markers: dict[str, str] = {}  # marker key -> job id whose result was fetched
        self._succeeded: set[str] = set()  # job ids the backend reported succeeded

    def _record(self, cfg: FanoutConfig, i: int) -> dict:
        key = status_key(cfg, i)
        if key not in self._records:
            if self.backend.exists(key):
                try:
                    rec = json.loads(self.backend.get_object(key))
                except (ValueError, UnicodeDecodeError) as e:
                    raise ShardError(f"unreadable status record {key}: {e}") from e
            else:
                rec = {"run_id": cfg.run_id, "shard": i, "shards": cfg.shards, "attempts": []}
            if (
                not isinstance(rec, dict)
                or rec.get("run_id") != cfg.run_id
                or rec.get("shard") != i
                or rec.get("shards") != cfg.shards
                or not isinstance(rec.get("attempts"), list)
                or any(not isinstance(a, dict) or "outcome" not in a for a in rec["attempts"])
            ):
                raise ShardError(f"invalid status record {key}: identity or attempts do not match this shard")
            for a in rec["attempts"]:
                if a["outcome"] == "running":
                    a.update(outcome="abandoned", reason="the orchestrator stopped before this attempt ended")
            self._records[key] = rec
        return self._records[key]

    def _save(self, cfg: FanoutConfig, i: int) -> None:
        self.backend.put_object(status_key(cfg, i), json.dumps(self._records[status_key(cfg, i)], indent=1).encode())

    def _describe(self, job_id: str) -> dict:
        describe = getattr(self.backend, "describe", None)
        if describe is None:
            return {}
        try:
            return describe(job_id) or {}
        except Exception:  # optional metadata must not lose a submitted job
            return {}

    def _end(self, job_id: str, outcome: str, stage: str | None = None, reason: str | None = None) -> None:
        self._close(*self._open.pop(job_id), outcome, stage, reason)

    def _close(
        self, cfg: FanoutConfig, i: int, a: dict, t0: float, outcome: str, stage: str | None, reason: str | None
    ) -> None:
        a.update(
            outcome=outcome,
            failed_stage=stage,
            reason=settings.redact(reason) if reason else None,
            finished_at=int(time.time()),
            wall_seconds=round(time.monotonic() - t0, 3),
        )
        self._save(cfg, i)

    def submit(self, cfg: FanoutConfig, i: int) -> str:
        rec = self._record(cfg, i)
        a = {
            "attempt": len(rec["attempts"]) + 1,
            "job_id": None,
            "submitted_at": int(time.time()),
            "finished_at": None,
            "wall_seconds": None,
            "resources": None,
            "states": [],
            "outcome": "running",
            "failed_stage": None,
            "reason": None,
        }
        rec["attempts"].append(a)
        t0 = time.monotonic()
        try:
            job_id = self.backend.submit(cfg, i)
        except Exception as e:
            self._close(cfg, i, a, t0, "failed", "submit", str(e))
            raise
        a.update(job_id=job_id, resources=self._describe(job_id).get("resources"))
        self._open[job_id] = (cfg, i, a, t0)
        self._save(cfg, i)
        return job_id

    def poll(self, job_id: str) -> str:
        try:
            state = self.backend.poll(job_id)
        except Exception as e:
            if job_id in self._open:
                self._end(job_id, "failed", "run", str(e))
            raise
        if job_id not in self._open:
            return state
        cfg, i, a, _ = self._open[job_id]
        info = self._describe(job_id)
        native = info.get("state") or state
        changed = not a["states"] or a["states"][-1]["state"] != native
        if changed:
            a["states"].append({"state": native, "at": int(time.time())})
        if state == FAILED:
            self._end(job_id, "failed", "run", info.get("reason") or f"job ended in state {native}")
        elif state == SUCCEEDED:
            self._succeeded.add(job_id)
            if changed:
                self._save(cfg, i)
        elif state != RUNNING:
            self._end(job_id, "failed", "run", f"backend returned unknown state {state!r}")
        elif changed:
            self._save(cfg, i)
        return state

    def fetch_result(self, job_id: str, prefix: str) -> None:
        try:
            self.backend.fetch_result(job_id, prefix)
            # The copy can succeed and still not be a shard dataset. Check before noting the done marker, or
            # cancel_open records the missing meta file as a marker failure and drops the path from the reason.
            if job_id in self._open:
                cfg, i, _, _ = self._open[job_id]
                require_shard_artifacts(self.backend, cfg, i)
        except Exception as e:
            if job_id in self._open:
                self._end(job_id, "failed", "fetch", str(e))
            raise
        if job_id in self._open:
            cfg, i, _, _ = self._open[job_id]
            self._markers[marker_key(cfg, i)] = job_id

    def put_object(self, key: str, data: bytes) -> None:
        job_id = self._markers.pop(key, None)
        try:
            self.backend.put_object(key, data)
        except Exception as e:
            if job_id:
                self._end(job_id, "failed", "marker", str(e))
            raise
        if job_id:
            self._end(job_id, "succeeded")

    def get_object(self, key: str) -> bytes:
        return self.backend.get_object(key)

    def exists(self, key: str) -> bool:
        return self.backend.exists(key)

    def list_objects(self, prefix: str) -> list[str]:
        return self.backend.list_objects(prefix)

    def cancel_open(self) -> list[str]:
        """The wrapped backend's `cancel_open`, then every attempt still open is recorded as failed: at run if its job
        was cancelled, at fetch or marker if the job had succeeded and the fan-out stopped before storing its result."""
        left = self.backend.cancel_open()
        for job_id in [j for j in self._open if j not in left]:
            if job_id in self._succeeded:
                stage = "marker" if job_id in self._markers.values() else "fetch"
                self._end(job_id, "failed", stage, "the fan-out stopped before this result was stored")
            else:
                self._end(job_id, "failed", "run", "cancelled when the fan-out stopped")
        return left
