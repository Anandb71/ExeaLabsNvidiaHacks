"""In-process fakes of the two Nebius APIs that arrival.jobs.nebius talks to, for offline tests (C1).

FakeS3 serves path-style S3 on 127.0.0.1: PUT, GET and HEAD object and ListObjectsV2 with continuation tokens (3 keys
a page, so paging is always exercised). It refuses a request whose Authorization header does not verify against its
one key pair, whose credential scope is not <day>/<region>/s3/aws4_request, whose signed headers leave out host,
x-amz-date or x-amz-content-sha256, or whose x-amz-content-sha256 is not the hash of the body. The signature is
recomputed from the decoded path and query, re-encoded the way S3 canonicalizes them, so a client that sends or
signs a key encoded any other way fails.

FakeJobs serves POST /ai/v1/jobs, GET /ai/v1/jobs/{id} and POST /ai/v1/jobs/cancel with the field names of the REST
reference, checks the bearer token, and refuses unknown fields, a missing parentId or image, and an environment
entry without exactly one of value and mysteryboxSecret. A created job "runs its container" as a local subprocess
(containerCommand plus args split on whitespace, in `workdir`, with the spec's environment and MysteryBox references
resolved from `secrets`). GET walks PROVISIONING, IMAGE_PULLING, then RUNNING until the process exits, then COMPLETED
or FAILED by exit code.

What it cannot check: Nebius's own validation (name rules, quotas, platform and preset pairs, image pulls), the
real enum spelling on the wire, MysteryBox payload lookup, how the real service splits `args`, preemption, timing,
and anything about Object Storage beyond the four S3 calls above.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.sax.saxutils import escape

from arrival.jobs.nebius import sigv4

S3_XMLNS = "http://s3.amazonaws.com/doc/2006-03-01/"
AUTH_RE = re.compile(
    r"AWS4-HMAC-SHA256 Credential=([^/]+)/(\d{8})/([^/]+)/s3/aws4_request, "
    r"SignedHeaders=([a-z0-9;-]+), Signature=([0-9a-f]{64})$"
)

# documented request fields (docs.nebius.com/rest-api/ai/v1/jobs/create)
REQUEST_FIELDS = {"metadata", "spec", "dryRun"}
METADATA_FIELDS = {"parentId", "name", "id", "labels", "resourceVersion"}
SPEC_FIELDS = {
    "image",
    "containerCommand",
    "args",
    "workingDir",
    "environmentVariables",
    "platform",
    "preset",
    "timeout",
    "restartAttempts",
    "disk",
    "volumes",
    "subnetId",
    "ports",
    "injectedFiles",
    "sshAuthorizedKeys",
    "publicIp",
    "preemptible",
    "shmSizeBytes",
    "pricingModel",
    "registryCredentials",
}
DISK_TYPES = {"UNSPECIFIED", "NETWORK_SSD", "NETWORK_HDD", "NETWORK_SSD_NON_REPLICATED", "NETWORK_SSD_IO_M3"}


class _Server:
    def __init__(self, handler):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.fake = self
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _send(self, status: int, body: bytes = b"", ctype: str = "application/xml"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


# ---- S3 ----


class _S3Handler(_Handler):
    def _error(self, status: int, code: str):
        self._send(status, f'<?xml version="1.0"?><Error><Code>{code}</Code></Error>'.encode())

    def _route(self):
        fake: FakeS3 = self.server.fake
        body = self._body()
        raw_path, _, raw_query = self.path.partition("?")
        problem = fake.verify(self.command, raw_path, raw_query, self.headers, body)
        if problem:
            fake.rejected.append(problem)
            return self._error(403, problem)
        parts = raw_path.lstrip("/").split("/", 1)
        if parts[0] != fake.bucket:
            return self._error(404, "NoSuchBucket")
        key = urllib.parse.unquote(parts[1]) if len(parts) > 1 else ""
        if self.command == "PUT" and key:
            fake.objects[key] = body
            return self._send(200)
        if self.command in ("GET", "HEAD") and key:
            if key not in fake.objects:
                return self._error(404, "NoSuchKey")
            return self._send(200, fake.objects[key], "application/octet-stream")
        if self.command == "GET" and not key:
            q = dict(urllib.parse.parse_qsl(raw_query, keep_blank_values=True))
            if q.get("list-type") != "2":
                return self._error(400, "InvalidArgument")
            return self._send(200, fake.list_xml(q.get("prefix", ""), q.get("continuation-token")))
        return self._error(405, "MethodNotAllowed")

    do_GET = do_PUT = do_HEAD = _route


class FakeS3(_Server):
    def __init__(self, bucket: str, key_id: str, secret: str, region: str, page: int = 3):
        self.bucket, self.key_id, self.secret, self.region, self.page = bucket, key_id, secret, region, page
        self.objects: dict[str, bytes] = {}
        self.rejected: list[str] = []
        super().__init__(_S3Handler)

    def verify(self, method, raw_path, raw_query, headers, body) -> str | None:
        m = AUTH_RE.match(headers.get("Authorization") or "")
        if not m:
            return "AccessDenied"
        key_id, day, region, signed, signature = m.groups()
        if key_id != self.key_id:
            return "InvalidAccessKeyId"
        if region != self.region or day != (headers.get("x-amz-date") or "")[:8]:
            return "AuthorizationHeaderMalformed"
        names = signed.split(";")
        if not {"host", "x-amz-date", "x-amz-content-sha256"} <= set(names):
            return "AccessDenied"
        if headers.get("x-amz-content-sha256") != hashlib.sha256(body).hexdigest():
            return "XAmzContentSHA256Mismatch"
        query = urllib.parse.parse_qsl(raw_query, keep_blank_values=True)
        path = urllib.parse.quote(urllib.parse.unquote(raw_path), safe="/-_.~")  # S3 canonicalizes the decoded key
        auth, _ = sigv4(
            method,
            path,
            query,
            {n: headers.get(n) or "" for n in names},
            headers["x-amz-content-sha256"],
            key_id,
            self.secret,
            region,
            "s3",
            headers["x-amz-date"],
        )
        return None if auth.endswith(f"Signature={signature}") else "SignatureDoesNotMatch"

    def list_xml(self, prefix: str, token: str | None) -> bytes:
        keys = sorted(k for k in self.objects if k.startswith(prefix) and (token is None or k > token))
        page, more = keys[: self.page], len(keys) > self.page
        items = "".join(f"<Contents><Key>{escape(k)}</Key><Size>{len(self.objects[k])}</Size></Contents>" for k in page)
        tail = f"<NextContinuationToken>{escape(page[-1])}</NextContinuationToken>" if more else ""
        return (
            f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="{S3_XMLNS}">'
            f"<Name>{self.bucket}</Name><Prefix>{escape(prefix)}</Prefix><KeyCount>{len(page)}</KeyCount>"
            f"{items}<IsTruncated>{'true' if more else 'false'}</IsTruncated>{tail}</ListBucketResult>"
        ).encode()


# ---- Serverless AI Jobs ----


class _JobsHandler(_Handler):
    def _json(self, status: int, obj: dict):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _route(self):
        fake: FakeJobs = self.server.fake
        body = self._body()
        if self.headers.get("Authorization") != f"Bearer {fake.token}":
            return self._json(401, {"code": 16, "message": "unauthenticated"})
        fake.requests.append((self.command, self.path, body))
        try:
            payload = json.loads(body) if body else None
            if self.command == "POST" and self.path == "/ai/v1/jobs":
                return self._json(200, fake.create(payload))
            if self.command == "POST" and self.path == "/ai/v1/jobs/cancel":
                return self._json(200, fake.cancel(payload))
            m = re.fullmatch(r"/ai/v1/jobs/([A-Za-z0-9_-]+)", self.path)
            if self.command == "GET" and m:
                return self._json(200, fake.get(m.group(1)))
        except (KeyError, ValueError) as e:
            return self._json(400 if isinstance(e, ValueError) else 404, {"code": 3, "message": str(e)})
        return self._json(404, {"code": 5, "message": f"no route {self.command} {self.path}"})

    do_GET = do_POST = _route


class FakeJobs(_Server):
    """`fail_first`: shards whose first job ends FAILED without running. `hold`: shards whose jobs stay RUNNING and
    run nothing until cancelled. `states`: shard -> a state to report once the job is past provisioning. `silent`:
    shards whose jobs run nothing and answer GET with no state at all until cancelled."""

    def __init__(
        self, token: str, project: str, secrets: dict, workdir, logdir, fail_first=(), hold=(), states=None, silent=()
    ):
        self.token, self.project, self.secrets, self.workdir, self.logdir = token, project, secrets, workdir, logdir
        self.fail_first, self.hold, self.states = set(fail_first), set(hold), dict(states or {})
        self.silent = set(silent)
        self.requests: list[tuple[str, str, bytes]] = []
        self.created: list[tuple[int, str]] = []  # (shard, job id) in order
        self.cancelled: list[str] = []
        self.jobs: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        super().__init__(_JobsHandler)

    @staticmethod
    def _check_fields(obj, allowed: set, where: str):
        if not isinstance(obj, dict):
            raise ValueError(f"{where} must be an object")
        unknown = set(obj) - allowed
        if unknown:
            raise ValueError(f"unknown field(s) in {where}: {sorted(unknown)}")

    def create(self, req) -> dict:
        self._check_fields(req, REQUEST_FIELDS, "request")
        md, spec = req.get("metadata"), req.get("spec")
        self._check_fields(md, METADATA_FIELDS, "metadata")
        self._check_fields(spec, SPEC_FIELDS, "spec")
        if md.get("parentId") != self.project:
            raise ValueError("metadata.parentId is required and must be the project")
        if not spec.get("image"):
            raise ValueError("spec.image is required")
        if "disk" in spec:
            self._check_fields(spec["disk"], {"type", "sizeBytes"}, "spec.disk")
            if spec["disk"].get("type") not in DISK_TYPES or not str(spec["disk"].get("sizeBytes", "")).isdigit():
                raise ValueError("spec.disk needs a known type and sizeBytes as a decimal string")
        env = {}
        for e in spec.get("environmentVariables") or []:
            self._check_fields(e, {"name", "value", "mysteryboxSecret"}, "environmentVariables[]")
            if not e.get("name") or (("value" in e) == ("mysteryboxSecret" in e)):
                raise ValueError("an environment variable needs a name and exactly one of value, mysteryboxSecret")
            if "mysteryboxSecret" in e:
                ref = e["mysteryboxSecret"]
                self._check_fields(ref, {"secretId", "versionId"}, "mysteryboxSecret")
                env[e["name"]] = self.secrets[ref["secretId"]][e["name"]]
            else:
                env[e["name"]] = e["value"]
        args = (spec.get("args") or "").split()
        shard = int(args[args.index("--shard") + 1].split("/")[0])
        with self._lock:
            job_id = f"aijob-{next(self._ids):04d}"
            first = not any(s == shard for s, _ in self.created)
            self.created.append((shard, job_id))
            job = {"metadata": {**md, "id": job_id}, "spec": spec, "gets": 0, "proc": None, "final": None}
            self.jobs[job_id] = job
        if shard in self.fail_first and first:
            job["final"] = ("FAILED", {"code": "ExitCode", "message": "container exited with code 3"})
        elif shard not in self.hold | self.silent:
            cmd = sys.executable if spec.get("containerCommand") == "python" else spec.get("containerCommand")
            log = open(os.path.join(self.logdir, f"{job_id}.log"), "wb")
            # the container sees only the spec's NEBIUS_* variables, never the test process's own
            base = {k: v for k, v in os.environ.items() if not k.startswith("NEBIUS_")}
            job["proc"] = subprocess.Popen(
                [cmd, *args], cwd=self.workdir, env={**base, **env}, stdout=log, stderr=subprocess.STDOUT
            )
            job["log"] = log
        job["shard"] = shard
        return {"id": f"aiop-{job_id}", "resourceId": job_id, "description": "create job"}

    def _state(self, job) -> tuple[str | None, dict | None]:
        if job["final"]:
            return job["final"]
        if job["shard"] in self.silent:
            return None, None
        job["gets"] += 1
        if job["gets"] == 1:
            return "PROVISIONING", None
        if job["gets"] == 2:
            return "IMAGE_PULLING", None
        if job["shard"] in self.states:
            return self.states[job["shard"]], None
        proc = job["proc"]
        if proc is None or proc.poll() is None:
            return "RUNNING", None
        job["log"].close()
        job["final"] = (
            ("COMPLETED", None)
            if proc.returncode == 0
            else ("FAILED", {"code": "ExitCode", "message": f"container exited with code {proc.returncode}"})
        )
        return job["final"]

    def get(self, job_id: str) -> dict:
        job = self.jobs[job_id]
        state, details = self._state(job)
        status = {"state": state, **({"stateDetails": details} if details else {})} if state else {}
        return {"metadata": job["metadata"], "spec": job["spec"], "status": status}

    def cancel(self, req) -> dict:
        self._check_fields(req, {"id"}, "cancel request")
        job = self.jobs[req["id"]]
        if job["proc"] and job["proc"].poll() is None:
            job["proc"].kill()
            job["proc"].wait()
        job["final"] = ("CANCELLED", None)
        self.cancelled.append(req["id"])
        return {"id": f"aiop-cancel-{req['id']}", "resourceId": req["id"], "description": "cancel job"}

    def close(self):
        for job in self.jobs.values():
            if job["proc"] and job["proc"].poll() is None:
                job["proc"].kill()
                job["proc"].wait()
            if job.get("log"):
                job["log"].close()
        super().close()
