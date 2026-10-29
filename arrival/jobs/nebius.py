"""Nebius backend for `run_fanout` (C1): each shard runs as a Nebius Serverless AI Job, and everything the run keeps
(the run manifest, shard outputs, done-markers) lives in one Nebius Object Storage bucket.

Two halves, stdlib only (urllib, hmac, xml), like the Token Factory client:

- `NebiusBackend` runs on the machine that calls `run_fanout`. `submit` creates a job through the REST API
  (POST /ai/v1/jobs) from the arrival image, `poll` reads its state (GET /ai/v1/jobs/{id}) and `cancel` stops it
  (POST /ai/v1/jobs/cancel). The object calls go to the bucket's S3-compatible API, signed with Signature Version 4.
- `python -m arrival.jobs.nebius run-shard` is the job's command inside the container. It runs `arrival sim` for one
  shard on the job's own disk and uploads the shard folder under a staging prefix unique to that job. `fetch_result`
  copies the staged files to `<run>/shards/<i>of<n>/`, then run_fanout writes the done-marker last, as with
  LocalBackend, so a failed or half-uploaded attempt never reaches the shard prefix.

Tested only against in-process fakes of the job API and of S3 (tests/fake_nebius.py). It has never run against
Nebius: no job was created and no bucket was touched. Request and response shapes follow docs.nebius.com as read on
2026-10-21; the pages are listed in docs/REFERENCE.md."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from arrival import config as settings
from arrival.jobs.fanout import FAILED, RUNNING, SUCCEEDED, FanoutConfig
from arrival.jobs.shards import ShardError

JOB_SETTINGS = ("NEBIUS_IAM_TOKEN", "NEBIUS_PROJECT_ID", "NEBIUS_JOB_IMAGE", "NEBIUS_JOB_S3_SECRET_ID")
STORE_SETTINGS = ("NEBIUS_BUCKET", "NEBIUS_ACCESS_KEY_ID", "NEBIUS_SECRET_ACCESS_KEY")
SECRET_ENV = ("NEBIUS_ACCESS_KEY_ID", "NEBIUS_SECRET_ACCESS_KEY")  # injected into the job from MysteryBox
DISK_BYTES = str(64 * 1024**3)  # the docs ask for at least 20 GiB on CPU platforms; left unset it would be 250 GiB

# JobStatus.State from the REST reference. CANCELLING and DELETING can no longer end in COMPLETED.
JOB_STATES = {
    "STATE_UNSPECIFIED": RUNNING,
    "PROVISIONING": RUNNING,
    "STARTING": RUNNING,
    "IMAGE_PULLING": RUNNING,
    "RUNNING": RUNNING,
    "COMPLETED": SUCCEEDED,
    "FAILED": FAILED,
    "CANCELLED": FAILED,
    "ERROR": FAILED,
    "CANCELLING": FAILED,
    "DELETING": FAILED,
}
FINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}


class NebiusNotConfigured(RuntimeError):
    pass


def require(*names: str) -> dict[str, str]:
    """The values of the named settings. Raises NebiusNotConfigured, naming every missing one, before anything is
    submitted."""
    values = {n: settings.get(n) for n in names}
    missing = [n for n, v in values.items() if not v]
    if missing:
        raise NebiusNotConfigured(
            f"set {', '.join(missing)} for the Nebius backend (see .env.example and "
            f"docs/REFERENCE.md); nothing was submitted"
        )
    return values


def _seconds(name: str) -> float:
    raw = settings.get(name)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise NebiusNotConfigured(f"{name} must be a number of seconds above 0, not {raw!r}")
    return value


def _http(
    method: str, url: str, body: bytes | None = None, headers: dict | None = None, timeout: float = 60.0
) -> tuple[int, bytes]:
    """(status, body); an HTTP error status is returned, not raised, and a network failure is a ShardError."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        raise ShardError(f"{method} {url.split('?')[0]}: {getattr(e, 'reason', e)}") from e


# ---- Object Storage (S3 API, Signature Version 4) ----


def _quote(s: str, safe: str = "-_.~") -> str:
    return urllib.parse.quote(s, safe=safe)


def canonical_query(pairs) -> str:
    return "&".join(f"{k}={v}" for k, v in sorted((_quote(k), _quote(v)) for k, v in pairs))


def sigv4(
    method: str,
    path: str,
    query,
    headers: dict,
    payload_hash: str,
    key_id: str,
    secret: str,
    region: str,
    service: str,
    amz_date: str,
) -> tuple[str, str]:
    """Signature Version 4 over every header in `headers`. `path` is the URI path exactly as sent (already
    percent-encoded, as S3 wants it not encoded twice), `query` a list of (name, value) pairs. Returns the
    Authorization value and the canonical request."""
    h = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    signed = ";".join(sorted(h))
    canonical = "\n".join(
        [method, path, canonical_query(query), *(f"{k}:{h[k]}" for k in sorted(h)), "", signed, payload_hash]
    )
    day = amz_date[:8]
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = ("AWS4" + secret).encode()
    for part in (day, region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    return f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, SignedHeaders={signed}, Signature={signature}", canonical


class ObjectStore:
    """Path-style S3 client for one bucket: https://storage.<region>.nebius.cloud/<bucket>/<key>."""

    def __init__(self, endpoint: str, region: str, bucket: str, key_id: str, secret: str):
        self.endpoint, self.region, self.bucket = endpoint.rstrip("/"), region, bucket
        self.host = urllib.parse.urlsplit(self.endpoint).netloc
        self._key_id, self._secret = key_id, secret

    @classmethod
    def from_settings(cls) -> ObjectStore:
        """The bucket named by the NEBIUS_* settings; the endpoint defaults to the region's storage endpoint."""
        v = require(*STORE_SETTINGS)
        region = settings.get("NEBIUS_REGION")
        endpoint = settings.get("NEBIUS_S3_ENDPOINT") or f"https://storage.{region}.nebius.cloud"
        return cls(endpoint, region, v["NEBIUS_BUCKET"], v["NEBIUS_ACCESS_KEY_ID"], v["NEBIUS_SECRET_ACCESS_KEY"])

    def _call(self, method: str, key: str = "", query=(), body: bytes = b"") -> tuple[int, bytes]:
        path = f"/{self.bucket}" + (f"/{_quote(key, safe='/-_.~')}" if key else "")
        payload_hash = hashlib.sha256(body).hexdigest()
        amz_date = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        headers = {"host": self.host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
        auth, _ = sigv4(
            method, path, query, headers, payload_hash, self._key_id, self._secret, self.region, "s3", amz_date
        )
        qs = canonical_query(query)
        url = self.endpoint + path + (f"?{qs}" if qs else "")
        send = {"x-amz-content-sha256": payload_hash, "x-amz-date": amz_date, "Authorization": auth}
        return _http(method, url, body if method == "PUT" else None, send)

    def _check(self, method: str, key: str, status: int, body: bytes) -> None:
        if status != 200:
            code = re.search(rb"<Code>([^<]+)</Code>", body or b"")
            raise ShardError(
                settings.redact(
                    f"object storage {method} {key or '/'}: HTTP {status}"
                    + (f" {code.group(1).decode(errors='replace')}" if code else "")
                )
            )

    def put(self, key: str, data: bytes) -> None:
        self._check("PUT", key, *self._call("PUT", key, body=data))

    def get(self, key: str) -> bytes:
        status, body = self._call("GET", key)
        self._check("GET", key, status, body)
        return body

    def exists(self, key: str) -> bool:
        status, body = self._call("HEAD", key)
        if status == 404:
            return False
        self._check("HEAD", key, status, body)
        return True

    def list(self, prefix: str) -> list[str]:
        """Every key under `prefix` (ListObjectsV2, following continuation tokens)."""
        keys, token = [], None
        while True:
            query = [("list-type", "2"), ("prefix", prefix)] + ([("continuation-token", token)] if token else [])
            status, body = self._call("GET", query=query)
            self._check("LIST", prefix, status, body)
            truncated, token = False, None
            for el in ElementTree.fromstring(body):
                tag = el.tag.rsplit("}", 1)[-1]
                if tag == "Contents":
                    keys += [c.text for c in el if c.tag.rsplit("}", 1)[-1] == "Key"]
                elif tag == "IsTruncated":
                    truncated = el.text == "true"
                elif tag == "NextContinuationToken":
                    token = el.text
            if not truncated:
                return sorted(keys)
            if not token:
                raise ShardError(f"object storage LIST {prefix}: truncated listing without a continuation token")


# ---- Serverless AI Jobs ----


class NebiusBackend:
    """`Backend` for run_fanout: shards as Serverless AI Jobs, storage in one Object Storage bucket. Settings come
    from arrival.config (see .env.example); a missing one raises NebiusNotConfigured naming it, before any request.

    `poll` cancels a job and reports it failed (so run_fanout retries it up to max_attempts) when it has not finished
    NEBIUS_MAX_JOB_SECONDS after it was created, or has reported no state for NEBIUS_MAX_NO_STATE_SECONDS in a row.
    `clock` is the monotonic clock those limits read (a test hook)."""

    def __init__(self, store: ObjectStore | None = None, *, timeout: str = "3600s", clock=None):
        v = require(*JOB_SETTINGS, *(() if store else STORE_SETTINGS))
        self.store = store or ObjectStore.from_settings()
        self.api = settings.get("NEBIUS_API_URL").rstrip("/")
        self._token = v["NEBIUS_IAM_TOKEN"]
        self.project, self.image = v["NEBIUS_PROJECT_ID"], v["NEBIUS_JOB_IMAGE"]
        self.s3_secret = v["NEBIUS_JOB_S3_SECRET_ID"]
        self.platform, self.preset = settings.get("NEBIUS_JOB_PLATFORM"), settings.get("NEBIUS_JOB_PRESET")
        self.subnet = settings.get("NEBIUS_JOB_SUBNET_ID")
        self.timeout = timeout  # the API allows 1 h to 168 h
        self.max_job_seconds = _seconds("NEBIUS_MAX_JOB_SECONDS")
        self.max_no_state_seconds = _seconds("NEBIUS_MAX_NO_STATE_SECONDS")
        if self.max_job_seconds <= float(timeout.rstrip("s")):
            raise NebiusNotConfigured(
                f"NEBIUS_MAX_JOB_SECONDS ({self.max_job_seconds:g}) must be longer than the job "
                f"timeout ({timeout}), or it would cut off a shard the service still allows"
            )
        self.clock = clock or time.monotonic
        self.submissions: list[tuple[int, int]] = []  # (shard, attempt number) in order
        self.jobs: dict[str, dict] = {}  # job id -> shard, name, staging prefix, last state, clock readings, reason

    def _api(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        status, data = _http(method, self.api + path, body, headers)
        if status != 200:
            raise ShardError(
                settings.redact(f"Nebius API {method} {path}: HTTP {status} {data[:300].decode(errors='replace')}")
            )
        try:  # a proxy's HTML page with HTTP 200 is a failed call, like any other: callers catch ShardError
            reply = json.loads(data or b"{}")
        except ValueError:
            reply = None
        if not isinstance(reply, dict):
            raise ShardError(
                settings.redact(
                    f"Nebius API {method} {path}: the HTTP 200 reply is not a JSON object: "
                    f"{data[:300].decode(errors='replace')}"
                )
            )
        return reply

    def job_spec(self, cfg: FanoutConfig, i: int, name: str, staging: str) -> dict:
        """The CreateJobRequest body. The access key pair reaches the job only as MysteryBox references."""
        args = ["-m", "arrival.jobs.nebius", "run-shard", "--dest", staging, "--", *cfg.sim_args(i)]
        bad = [a for a in args if not a or any(ch.isspace() for ch in a)]
        if bad:
            raise ShardError(f"job arguments cannot be empty or contain spaces (the API takes one args string): {bad}")
        env = [
            {"name": "NEBIUS_BUCKET", "value": self.store.bucket},
            {"name": "NEBIUS_S3_ENDPOINT", "value": self.store.endpoint},
            {"name": "NEBIUS_REGION", "value": self.store.region},
        ]
        env += [{"name": n, "mysteryboxSecret": {"secretId": self.s3_secret}} for n in SECRET_ENV]
        spec = {
            "image": self.image,
            "containerCommand": "python",
            "args": " ".join(args),
            "environmentVariables": env,
            "platform": self.platform,
            "preset": self.preset,
            "timeout": self.timeout,
            "disk": {"type": "NETWORK_SSD", "sizeBytes": DISK_BYTES},
        }
        if self.subnet:
            spec["subnetId"] = self.subnet
        return {"metadata": {"parentId": self.project, "name": name}, "spec": spec}

    def submit(self, cfg: FanoutConfig, i: int) -> str:
        n = sum(1 for s, _ in self.submissions if s == i) + 1
        self.submissions.append((i, n))
        tag = uuid.uuid4().hex[:8]  # a fresh staging prefix per attempt, so a retry never mixes with a dead one
        name = re.sub(r"[^a-z0-9-]", "-", f"arrival-{tag}-{i}of{cfg.shards}-{cfg.run_id}".lower())[:63].rstrip("-")
        staging = f"{cfg.run_id}/attempts/{i}of{cfg.shards}/{tag}"
        op = self._api("POST", "/ai/v1/jobs", self.job_spec(cfg, i, name, staging))
        if (op.get("status") or {}).get("code"):
            raise ShardError(settings.redact(f"Nebius refused job {name}: {op['status'].get('message')}"))
        job_id = op.get("resourceId")
        if not job_id:
            raise ShardError(f"Nebius create for job {name} returned no resourceId")
        now = self.clock()
        self.jobs[job_id] = {
            "shard": i,
            "shards": cfg.shards,
            "name": name,
            "staging": staging,
            "state": None,
            "details": None,
            "submitted": now,
            "seen": now,
            "reason": None,
        }
        return job_id

    def poll(self, job_id: str) -> str:
        status = self._api("GET", f"/ai/v1/jobs/{_quote(job_id)}").get("status") or {}
        state = status.get("state") or "STATE_UNSPECIFIED"
        if state not in JOB_STATES:
            raise ShardError(f"Nebius job {job_id} is in state {state!r}, which this client does not know")
        rec = self.jobs.get(job_id)
        if rec is None:
            return JOB_STATES[state]
        now = self.clock()
        if state != "STATE_UNSPECIFIED":
            rec["seen"] = now
        rec.update(state=state, details=status.get("stateDetails"))
        if JOB_STATES[state] != RUNNING:
            return JOB_STATES[state]
        if now - rec["submitted"] > self.max_job_seconds:
            limit = f"not finished {self.max_job_seconds:g} s after it was created"
        elif now - rec["seen"] > self.max_no_state_seconds:
            limit = f"reported no state for {self.max_no_state_seconds:g} s"
        else:
            return RUNNING
        try:
            self.cancel(job_id)
            rec["reason"] = f"{limit}; cancelled"
        except ShardError as e:  # cancel_open tries again when the command ends
            rec["reason"] = f"{limit}; cancel failed: {e}"
        return FAILED

    def describe(self, job_id: str) -> dict:
        """What the status record keeps for a job: its last Nebius state, why it failed, and the resources asked for."""
        rec = self.jobs[job_id]
        reason = rec["reason"]
        if reason is None and rec["state"] and JOB_STATES[rec["state"]] == FAILED:
            reason = rec["state"] + (f": {rec['details'].get('message')}" if rec["details"] else "")
        return {
            "state": rec["state"],
            "reason": reason,
            "resources": {
                "platform": self.platform,
                "preset": self.preset,
                "disk_bytes": int(DISK_BYTES),
                "timeout": self.timeout,
            },
        }

    def cancel(self, job_id: str) -> None:
        self._api("POST", "/ai/v1/jobs/cancel", {"id": job_id})
        self.jobs.get(job_id, {})["state"] = "CANCELLING"

    def cancel_open(self) -> list[str]:
        """Cancel every job this backend started that has not reached a final state (run_fanout does not cancel
        anything when it stops early). Returns the ids that could not be cancelled."""
        failed = []
        for job_id, rec in self.jobs.items():
            if rec["state"] not in FINAL_STATES and rec["state"] != "CANCELLING":
                try:
                    self.cancel(job_id)
                except ShardError:
                    failed.append(job_id)
        return failed

    def fetch_result(self, job_id: str, prefix: str) -> None:
        rec = self.jobs[job_id]
        if prefix.rsplit("/", 1)[-1] != f"{rec['shard']}of{rec['shards']}":
            raise ShardError(f"job {job_id} ran shard {rec['shard']}of{rec['shards']}, not {prefix}")
        src = rec["staging"] + "/"
        keys = self.store.list(src)
        if not keys:
            raise ShardError(f"job {job_id} reported success but uploaded nothing under {src}")
        for key in keys:
            self.put_object(f"{prefix}/{key[len(src) :]}", self.store.get(key))

    def put_object(self, key: str, data: bytes) -> None:
        self.store.put(key, data)

    def get_object(self, key: str) -> bytes:
        return self.store.get(key)

    def exists(self, key: str) -> bool:
        return self.store.exists(key)

    def list_objects(self, prefix: str) -> list[str]:
        return self.store.list(prefix.rstrip("/") + "/")  # the slash keeps 1of2 from matching 1of20


# ---- inside the job ----


def run_shard(dest: str, sim_args: list[str], store: ObjectStore | None = None) -> int:
    """Run one `arrival sim ... --shard i/n` on local disk, then upload its shard folder under `dest/`."""
    shard = sim_args[sim_args.index("--shard") + 1].replace("/", "of")
    store = store or ObjectStore.from_settings()
    with tempfile.TemporaryDirectory() as out:
        code = subprocess.call([sys.executable, "-m", "arrival", *sim_args, "--out", out])
        if code:
            return code
        src = Path(out) / "shards" / shard
        files = sorted(p for p in src.rglob("*") if p.is_file())
        if not files:
            print(f"run-shard: sim wrote nothing under {src}", file=sys.stderr)
            return 1
        for f in files:
            store.put(f"{dest}/{f.relative_to(src).as_posix()}", f.read_bytes())
    print(f"run-shard: uploaded {len(files)} files to {dest}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 5 or argv[:2] != ["run-shard", "--dest"] or argv[3] != "--":
        print("usage: python -m arrival.jobs.nebius run-shard --dest <prefix> -- <arrival sim args>", file=sys.stderr)
        return 2
    return run_shard(argv[2], argv[4:])


if __name__ == "__main__":
    sys.exit(main())
