"""Local pre-scanned room page and queued upload.

Standard library only. Single process, no authentication, binds 127.0.0.1 by default.
``POST /jobs`` stores the upload and returns 202. It never compiles or
generates inside the request, and this module does not import the pipeline.

Hardening (L4): unguessable job ids, upload size/type/content checks, safe filenames, a per-client rate limit,
a queue cap and a concurrent-upload cap, security headers with a hash-based CSP, generic error pages, redacted
error logs, and a ``public`` mode that lists only the caller's own jobs. There is still no authentication: put a
real front end (TLS, auth) in front before exposing it beyond a network you trust.
"""

from __future__ import annotations

import html
import json
import mmap
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from arrival.jobs.limits import MAX_UPLOAD_BYTES, LimitError, Limits
from arrival.jobs.queue import check_upload_size, count_active, create_job, read_job
from arrival.web import frames as frames_mod
from arrival.web import hardening as hd
from arrival.web.hardening import JOB_ID_RE

ID_RE = re.compile(r"^[a-z0-9_]+$")
SCAN_FILES = ("scan.json", "roomplan.json")  # what makes scans/<room> a room, in the order `sim` reads them
BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?', re.I)
NAME_RE = re.compile(rb'(?:^|;\s*)name="([^"]*)"', re.I)
FILENAME_RE = re.compile(rb'filename="([^"]*)"', re.I)
READ_CHUNK = 1 << 20
UPLOAD_DEADLINE = 600.0  # seconds for the whole body, on top of the per-read socket timeout
MAX_BOUNDARY = 200
DISK_MARGIN = 100 * 1024 * 1024
DRAIN_MAX = 2 * 1024 * 1024
LOOPBACK = ("127.0.0.1", "::1", "localhost")


class UploadError(ValueError):
    pass


def find_part(buf, boundary: bytes, field: str, optional: bool = False):
    """Scan a multipart body for form field ``field``; return (data offset, data length, filename).
    Raises UploadError on a missing closing boundary or a missing field (None for a missing ``optional`` one)."""
    delim = b"--" + boundary
    if buf.find(delim + b"--") < 0:
        raise UploadError("multipart body has no closing boundary (truncated?)")
    pos = buf.find(delim)
    while pos >= 0:
        start = pos + len(delim)
        if buf[start : start + 2] == b"--":
            break
        head_end = buf.find(b"\r\n\r\n", start)
        if head_end < 0:
            raise UploadError("malformed multipart part headers")
        headers = bytes(buf[start:head_end])
        nxt = buf.find(b"\r\n" + delim, head_end + 4)
        if nxt < 0:
            raise UploadError("multipart part has no terminating boundary")
        name = NAME_RE.search(headers)
        if name and name.group(1).decode("latin-1") == field:
            fn = FILENAME_RE.search(headers)
            return head_end + 4, nxt - (head_end + 4), fn.group(1).decode("utf-8", "replace") if fn else ""
        pos = nxt + 2
    if optional:
        return None
    raise UploadError(f"missing file field {field!r}")


CSS = """
:root { color-scheme: light dark;
  --fg:#18202a; --muted:#5d6874; --line:#d8dee5; --bg:#f4f6f9; --card:#fff; --card2:#eef2f6;
  --accent:#2563c9; --accent-fg:#fff; --ok:#17803d; --ok-bg:#e3f5e9; --bad:#b42318; --bad-bg:#fde8e6;
  --warn-bg:#fff4e0; --warn-fg:#7a4a00; --warn-line:#f0d199; --run:#b25e00; --shadow:0 1px 2px #0001, 0 4px 16px #0000000a; }
@media (prefers-color-scheme: dark) { :root {
  --fg:#e7ecf2; --muted:#98a3b0; --line:#2b3441; --bg:#0d1117; --card:#151b24; --card2:#1b2330;
  --accent:#5b9bff; --accent-fg:#06101f; --ok:#4cc77a; --ok-bg:#12301e; --bad:#ff8a80; --bad-bg:#3a1714;
  --warn-bg:#2d2208; --warn-fg:#f2c66d; --warn-line:#5a4413; --run:#f0a040; --shadow:none; } }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.55 -apple-system, system-ui, "Segoe UI", sans-serif; color:var(--fg); background:var(--bg); }
main { max-width: 1040px; margin: 0 auto; padding: 20px 16px 56px; }
header.top { display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:4px 0 18px; }
.brand { font-weight:700; font-size:20px; letter-spacing:.04em; color:var(--fg); text-decoration:none; }
.brand i { color:var(--accent); font-style:normal; }
.tag { color:var(--muted); font-size:14px; }
h1 { font-size:26px; margin:6px 0 4px; letter-spacing:-.01em; }
h2 { font-size:13px; margin:34px 0 10px; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); }
a { color:var(--accent); }
.muted { color:var(--muted); }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:12px 0; box-shadow:var(--shadow); }
.warn { background:var(--warn-bg); color:var(--warn-fg); border:1px solid var(--warn-line); border-radius:10px; padding:10px 14px; margin:12px 0; font-size:14px; }
.flow { display:flex; gap:8px; flex-wrap:wrap; margin:14px 0 4px; padding:0; list-style:none; counter-reset:f; }
.flow li { counter-increment:f; background:var(--card); border:1px solid var(--line); border-radius:999px; padding:4px 12px 4px 8px; font-size:13px; color:var(--muted); }
.flow li::before { content:counter(f); display:inline-block; width:20px; height:20px; line-height:20px; text-align:center; border-radius:50%; background:var(--accent); color:var(--accent-fg); font-size:12px; font-weight:700; margin-right:7px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:16px; }
.room { display:flex; flex-direction:column; padding:0; overflow:hidden; margin:0; transition:transform .12s, border-color .12s; }
.room:hover { transform:translateY(-2px); border-color:var(--accent); }
.thumb { aspect-ratio:16/9; background:var(--card2); display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:13px; overflow:hidden; }
.thumb img, .thumb video { width:100%; height:100%; object-fit:cover; display:block; }
.room .body { padding:14px 16px 16px; display:flex; flex-direction:column; gap:10px; flex:1; }
.room h3 { margin:0; font-size:18px; }
.room h3 a { color:var(--fg); text-decoration:none; }
.room h3 a::after { content:""; position:absolute; inset:0; }
.room { position:relative; }
.stats { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; }
.stat { background:var(--card2); border-radius:8px; padding:7px 9px; }
.stat b { display:block; font-size:18px; line-height:1.2; font-variant-numeric:tabular-nums; }
.stat span { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { display:inline-block; font-size:12px; padding:2px 9px; border-radius:999px; background:var(--card2); color:var(--muted); border:1px solid var(--line); white-space:nowrap; }
.chip.ok { background:var(--ok-bg); color:var(--ok); border-color:transparent; }
.chip.bad { background:var(--bad-bg); color:var(--bad); border-color:transparent; }
.chip.run { color:var(--run); }
.actions { display:flex; gap:8px; margin-top:auto; position:relative; z-index:1; }
.btn, button { display:inline-block; padding:7px 14px; border-radius:8px; border:1px solid var(--line); background:var(--card2); color:var(--fg); cursor:pointer; text-decoration:none; font:inherit; font-size:14px; }
.btn.primary, button.primary { background:var(--accent); color:var(--accent-fg); border-color:transparent; font-weight:600; }
.btn:hover, button:hover { filter:brightness(1.08); }
.btn:focus-visible, button:focus-visible, a:focus-visible, .drop:focus-within { outline:2px solid var(--accent); outline-offset:2px; }
.empty { text-align:center; padding:30px 18px; color:var(--muted); }
.empty b { display:block; color:var(--fg); font-size:16px; margin-bottom:4px; }
table { border-collapse: collapse; width: 100%; background: var(--card); border-radius:10px; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align: top; }
th { font-weight:600; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
tr:last-child td { border-bottom:0; }
.kv th { width:34%; text-transform:none; letter-spacing:0; font-size:13px; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.scroll { overflow-x: auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }
.drop { display:block; border:2px dashed var(--line); border-radius:12px; padding:22px; text-align:center; cursor:pointer; transition:border-color .12s, background .12s; }
.drop.over, .drop:hover { border-color:var(--accent); background:var(--card2); }
.drop input { max-width:100%; }
.upl { display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; align-items:center; }
progress { flex:1; min-width:140px; height:10px; accent-color:var(--accent); }
#msg { margin-top:10px; font-size:14px; min-height:1.2em; }
#msg.err { color:var(--bad); } #msg.ok { color:var(--ok); }
.stages { display:flex; gap:3px; min-width:150px; }
.stages i { flex:1; height:6px; border-radius:3px; background:var(--line); }
.stages i.done { background:var(--ok); }
.stages i.cur { background:var(--run); animation:pulse 1.2s ease-in-out infinite; }
.stages i.fail { background:var(--bad); }
@keyframes pulse { 50% { opacity:.35; } }
@media (prefers-reduced-motion: reduce) { .stages i.cur { animation:none; } .room { transition:none; } }
.stage-label { font-size:12px; color:var(--muted); margin-top:3px; }
.media { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px; }
.media figure { margin:0; } .media figcaption { font-size:12px; color:var(--muted); margin-top:4px; }
.media img, .media video { width:100%; border-radius:10px; border:1px solid var(--line); background:var(--card2); }
.crumb { margin:0 0 6px; font-size:14px; }
.mt { margin-top:8px; }
.drop.sub { padding:12px; margin-top:10px; font-size:14px; }
@media (max-width:560px) { h1 { font-size:22px; } .stats { grid-template-columns:repeat(3, 1fr); } }
"""

STAGES = ("normalize", "twin", "realism", "objects", "sim", "validate", "zip")  # arrival.jobs.worker stage order
PREVIEW_TYPES = {".png": "image/png", ".mp4": "video/mp4", ".gif": "image/gif"}
PREVIEW_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
UPLOADED_ASSET_RE = re.compile(r"(?:^|_)(job_[0-9a-f]{8,32})_sim(?:$|_)")

JS = """
(function () {
  var q = function (s) { return document.querySelector(s); };
  function refreshJobs() {
    return fetch('/', {headers: {Accept: 'text/html'}}).then(function (r) { return r.text(); }).then(function (t) {
      var doc = new DOMParser().parseFromString(t, 'text/html'), n = doc.getElementById('jobs'), o = q('#jobs');
      if (n && o) { o.innerHTML = n.innerHTML; watch(); }
    }).catch(function () {});
  }
  var timer = null;
  function watch() {
    if (timer) { clearInterval(timer); timer = null; }
    if (!document.querySelector('#jobs [data-active]')) return;
    timer = setInterval(function () { if (!document.hidden) refreshJobs(); }, 2500);
  }
  watch();
  document.addEventListener('click', function (ev) {
    var b = ev.target.closest ? ev.target.closest('button[data-job-action]') : null;
    if (!b) return;
    if (b.dataset.jobAction === 'delete' && !confirm('Delete this job and its dataset? This cannot be undone.')) return;
    b.disabled = true;
    fetch('/jobs/' + b.dataset.jobId, {method: 'DELETE', headers: {Accept: 'application/json'}}).then(refreshJobs, refreshJobs);
  });
  var form = q('#upload'); if (!form) return;
  var file = form.querySelector('input[type=file]'), drop = q('.drop'), msg = q('#msg'), bar = q('progress'), name = q('#fname');
  function say(t, c) { msg.textContent = t; msg.className = c || ''; }
  file.addEventListener('change', function () { name.textContent = file.files[0] ? file.files[0].name : 'Choose or drop a scan file'; });
  ['dragenter', 'dragover'].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (e) { drop.addEventListener(e, function () { drop.classList.remove('over'); }); });
  drop.addEventListener('drop', function (ev) { ev.preventDefault(); if (ev.dataTransfer.files.length) { file.files = ev.dataTransfer.files; file.dispatchEvent(new Event('change')); } });
  form.addEventListener('submit', function (ev) {
    if (!window.XMLHttpRequest || !window.FormData) return;
    ev.preventDefault();
    if (!file.files.length) return say('Pick a file first.', 'err');
    var x = new XMLHttpRequest(), btn = form.querySelector('button');
    x.open('POST', '/jobs'); x.setRequestHeader('Accept', 'application/json');
    x.upload.onprogress = function (e) { if (e.lengthComputable) { bar.hidden = false; bar.value = e.loaded / e.total; } };
    x.onload = function () {
      btn.disabled = false; bar.hidden = true;
      var d = {}; try { d = JSON.parse(x.responseText); } catch (e) {}
      if (x.status === 202) { say('Queued as ' + d.id + '. Run python -m arrival work to process it.', 'ok'); form.reset(); name.textContent = 'Choose or drop a scan file'; refreshJobs(); }
      else say('Upload failed (' + x.status + '): ' + (d.error || 'unexpected response'), 'err');
    };
    x.onerror = function () { btn.disabled = false; bar.hidden = true; say('Upload failed: server unreachable.', 'err'); };
    btn.disabled = true; say('Uploading...');
    x.send(new FormData(form));
  });
})();
"""

PLACEHOLDER_NOTE = (
    "Motion in this dataset is a scripted placeholder in simulation "
    "(arrival_generator = scripted_retarget). It is not physically valid, not "
    "MimicGen, and not a real-robot result."
)


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def room_scan(scans_dir: Path, room: str) -> tuple[dict, str] | None:
    """The room's scan and the file it came from: scans/<room>/scan.json, else its roomplan.json normalized to a
    bundle (fixtures, geometry_source). A phone capture lands with only roomplan.json and `sim` builds it (#417);
    listing only scan.json rooms left such a room off the page with its sim dataset and 404'd /rooms/<room>.
    None when the room has neither file."""
    for name in SCAN_FILES:
        path = Path(scans_dir) / room / name
        if not path.is_file():
            continue
        data = _read_json(path)
        if name == "roomplan.json":
            from arrival.scan import roomplan  # standard library plus the scan module; no pipeline import

            try:
                data = roomplan.to_bundle(data, room)
            except Exception:  # noqa: BLE001 - an unreadable export still gets its card, never a 500
                data = {"geometry_source": "roomplan"}
        return (data if isinstance(data, dict) else {}), name
    return None


def _page(title: str, body: str) -> bytes:
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta name='color-scheme' content='light dark'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head>"
        "<body><main><header class='top'><a class='brand' href='/'>ARRIVAL<i>.</i></a>"
        "<span class='tag'>scan a room, get robot training data</span></header>"
        f"{body}</main><script>{JS}</script></body></html>"
    )
    return doc.encode("utf-8")


def _stage_bar(status: dict) -> str:
    """Segmented progress for a job: one segment per worker stage, current one pulsing."""
    st, stage = status.get("status"), status.get("stage")
    idx = STAGES.index(stage) if stage in STAGES else -1
    segs = []
    for i, name in enumerate(STAGES):
        if st == "done" or i < idx:
            cls = "done"
        elif i == idx:
            cls = "fail" if st == "failed" else "cur"
        else:
            cls = ""
        segs.append(f"<i class='{cls}' title='{name}'></i>")
    label = {"done": "complete", "queued": "waiting for worker", "failed": f"failed at {stage or 'start'}"}.get(
        st, f"{stage or 'starting'} ({idx + 1}/{len(STAGES)})" if idx >= 0 else "starting"
    )
    prog = status.get("progress")
    if st == "running" and stage == "sim" and isinstance(prog, dict) and prog.get("total"):
        label = f"sim: {prog.get('done', 0)}/{prog['total']} episodes ({idx + 1}/{len(STAGES)})"
    return f"<div class='stages' role='img' aria-label='{esc(label)}'>{''.join(segs)}</div><div class='stage-label'>{esc(label)}</div>"


def _chip(text: str, kind: str = "") -> str:
    return f"<span class='chip {kind}'>{esc(text)}</span>"


def _pct(num, den) -> str:
    return (
        f"{100 * num / den:.0f}%" if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den else "n/a"
    )


def _fmt_list(values) -> str:
    if isinstance(values, (list, tuple)):
        return "[" + ", ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in values) + "]"
    return str(values)


class ArrivalServer(ThreadingHTTPServer):
    """The threaded HTTP server and the settings and shared state every request reads: folders, upload and queue limits,
    public mode, proxy trust, the rate limiter and the locks that serialise job creation and cap concurrent uploads."""

    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that is already being served; ask for exclusive use.
    allow_reuse_address = sys.platform != "win32"
    out_dir: Path
    scans_dir: Path
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    max_queued: int = 50  # queued + running jobs; 0 = unlimited
    public: bool = False  # list only the caller's own jobs
    behind_proxy: bool = False  # trust X-Forwarded-Proto / X-Forwarded-For from a reverse proxy
    limiter: hd.RateLimiter
    post_lock: threading.Lock
    heavy_slots: threading.BoundedSemaphore

    def server_bind(self):
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _csp(inline: bool) -> str:
    """Pages inline one stylesheet and one script, allowed by hash (no 'unsafe-inline'); JSON gets the empty policy."""
    if not inline:
        return "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    return (
        "default-src 'none'; "
        f"style-src {hd.csp_hash(CSS)}; script-src {hd.csp_hash(JS)}; "
        "img-src 'self' data:; media-src 'self'; connect-src 'self'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )


def _status_phrase(code: int) -> str:
    try:
        return HTTPStatus(code).phrase
    except ValueError:
        return "Error"


class Handler(BaseHTTPRequestHandler):
    """Every route of the local page: the room list and room pages, job status and deletion, dataset downloads, previews
    and the upload endpoint. Every reply carries the security headers."""

    server: ArrivalServer
    server_version = "Arrival"
    sys_version = ""  # no Python version in the Server header
    timeout = 30  # seconds per socket read; a stalled upload is dropped instead of pinning a thread

    # ---- helpers -------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        return Path(self.server.out_dir)

    @property
    def scans_dir(self) -> Path:
        return Path(self.server.scans_dir)

    def log_message(self, format, *args):  # quieter default logging: no access log, so no URLs or headers on disk
        pass

    def log_error(self, format, *args):  # only protocol errors and crashes, redacted
        try:
            sys.stderr.write("[arrival.web] " + hd.redact(format % args if args else format) + "\n")
        except Exception:  # noqa: BLE001 - logging must never break a request
            pass

    def version_string(self) -> str:
        return self.server_version

    @property
    def https(self) -> bool:
        headers = getattr(self, "headers", None)  # absent when the request line itself was malformed
        return bool(
            self.server.behind_proxy and headers is not None and headers.get("X-Forwarded-Proto", "").lower() == "https"
        )

    def _start(
        self,
        code: int,
        ctype: str,
        length: int,
        extra: dict | None = None,
        inline: bool = False,
        cache: str = "no-store",
    ):
        """Status line plus the headers every response carries; the caller ends the headers."""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        for k, v in hd.security_headers(_csp(inline), self.https).items():
            self.send_header(k, v)
        for k, v in (extra or {}).items():
            self.send_header(k, v)

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self._start(code, ctype, len(body), extra, inline=ctype.startswith("text/html"))
        self.end_headers()
        if getattr(self, "command", None) != "HEAD":
            self.wfile.write(body)

    def _send_file(self, f, size: int, ctype: str, extra: dict | None = None):
        """Stream an open file in READ_CHUNK pieces so a download never sits in memory whole."""
        self._start(200, ctype, size, extra)
        self.end_headers()
        if self.command != "HEAD":
            shutil.copyfileobj(f, self.wfile, READ_CHUNK)

    def _html(self, code: int, title: str, body: str, extra: dict | None = None):
        self._send(code, _page(title, body), "text/html; charset=utf-8", extra)

    def _json(self, code: int, obj, extra: dict | None = None):
        self._send(code, (json.dumps(obj, indent=2) + "\n").encode(), "application/json", extra)

    def _wants_json(self) -> bool:
        path = (getattr(self, "path", None) or "").split("?", 1)[0]
        headers = getattr(self, "headers", None)
        accept = headers.get("Accept", "") if headers is not None else ""
        return (
            getattr(self, "command", None) == "POST"
            or path.startswith("/jobs")
            or ("application/json" in accept and "text/html" not in accept)
        )

    def _error(self, code: int, msg: str | None = None, extra: dict | None = None):
        """A generic error page (HTML for browsers, JSON for the API and the capture app). Never a traceback."""
        msg = msg or _status_phrase(code)
        if self._wants_json():
            return self._json(code, {"error": msg}, extra)
        self._html(
            code,
            _status_phrase(code),
            f"<div class='card empty'><b>{code}</b>{esc(msg)}<p><a class='btn' href='/'>Back to rooms</a></p></div>",
            extra,
        )

    def send_error(self, code, message=None, explain=None):  # protocol-level errors from http.server
        self.close_connection = True
        if getattr(self, "request_version", "HTTP/0.9") == "HTTP/0.9":
            self.request_version = "HTTP/1.0"  # http.server would send no status line or headers at all
        try:
            self._error(int(code))
        except Exception:  # noqa: BLE001 - the socket may already be gone
            pass

    def _guard(self, fn):
        """Run a route; a crash becomes a generic 500 and a redacted stderr line, a dropped socket is ignored."""
        try:
            fn()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as e:  # noqa: BLE001
            self.log_error(
                "unhandled %s in %s %s", type(e).__name__, self.command, (self.path or "").split("?", 1)[0][:80]
            )
            self.close_connection = True
            try:
                self._error(500, "Internal server error")
            except Exception:  # noqa: BLE001
                pass

    def _not_found(self, msg: str = "Not found"):
        self._error(404, msg)

    def _client(self) -> str:
        return hd.client_ip(self.client_address[0], self.headers.get("X-Forwarded-For"), self.server.behind_proxy)

    def _owner(self) -> str | None:
        return hd.owner_of(hd.cookie_value(self.headers.get("Cookie")))

    def _rooms(self) -> list[str]:
        if not self.scans_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in self.scans_dir.iterdir()
            if p.is_dir() and any((p / f).is_file() for f in SCAN_FILES) and ID_RE.match(p.name)
        )

    def _scan(self, room: str) -> tuple[dict, str] | None:
        return room_scan(self.scans_dir, room)

    # ---- routing -------------------------------------------------------
    def do_GET(self):
        self._guard(self._route_get)

    do_HEAD = do_GET

    def _route_get(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/":
            return self._index()
        if path == "/healthz":  # liveness for a proxy or container healthcheck: no disk, no job data
            return self._json(200, {"status": "ok"})
        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "previews":
            return self._preview_file(parts[1])
        if len(parts) in (3, 4) and parts[0] == "jobs" and parts[2] == "realism":
            artifact = parts[3] if len(parts) == 4 else ""
            if not JOB_ID_RE.match(parts[1]) or artifact not in ("", "report", "preview"):
                return self._not_found()
            return self._job_realism(parts[1], artifact)
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "download":
            if not JOB_ID_RE.match(parts[1]):
                return self._not_found("Invalid id")
            return self._job_download(parts[1])
        if len(parts) == 2 and parts[0] in ("rooms", "download", "jobs"):
            ident = parts[1]
            if not (JOB_ID_RE if parts[0] == "jobs" else ID_RE).match(ident):
                return self._not_found("Invalid id")
            if parts[0] == "rooms":
                return self._room(ident)
            if parts[0] == "download":
                return self._download(ident)
            return self._job(ident)
        return self._not_found()

    def do_POST(self):
        def route():
            path = self.path.split("?", 1)[0].rstrip("/")
            if path != "/jobs":
                return self._not_found()
            return self._post_job()

        self._guard(route)

    def _method_not_allowed(self):
        self._error(405, extra={"Allow": "GET, HEAD, POST, DELETE"})

    def do_DELETE(self):
        def route():
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if len(parts) != 2 or parts[0] != "jobs" or not JOB_ID_RE.match(parts[1]):
                return self._not_found()
            return self._delete_job(parts[1])

        self._guard(route)

    do_PUT = do_PATCH = do_OPTIONS = _method_not_allowed

    # ---- pages ---------------------------------------------------------
    def _task_counts(self, room: str):
        tasks = _read_json(self.out_dir / "twins" / room / "tasks.json")
        if not tasks:
            return None
        return Counter(t.get("family", "?") for t in tasks.get("tasks", []))

    def _ds_stats(self, name: str) -> dict | None:
        """Headline numbers for out/datasets/<name>, or None when that dataset is absent."""
        ds = self.out_dir / "datasets" / name
        if not ds.is_dir():
            return None
        info = _read_json(ds / "meta" / "info.json")
        info = info if isinstance(info, dict) else {}
        rep = _read_json(self.out_dir / "reports" / f"{name}.json")
        rep = rep if isinstance(rep, dict) else {}
        succ = info.get("arrival_successes", rep.get("successes"))
        att = info.get("arrival_attempts", rep.get("attempts"))
        contacts = rep.get("human_contacts")
        if (
            contacts is None
            and isinstance(rep.get("families"), dict)
            and all(isinstance(f, dict) and "human_contacts" in f for f in rep["families"].values())
            and rep["families"]
        ):
            contacts = sum(f["human_contacts"] for f in rep["families"].values())
        return {
            "info": bool(info),
            "episodes": info.get("total_episodes"),
            "successes": succ,
            "attempts": att,
            "rate": _pct(succ, att),
            "valid": info.get("arrival_physically_valid") if info else None,
            "generator": info.get("arrival_generator"),
            "contacts": contacts,
            "fps": info.get("fps"),
            "families": info.get("arrival_families") or [],
            "report": rep,
        }

    def _preview(self, name: str, ext: str) -> str | None:
        f = f"{name}_{ext}"
        return f"/previews/{f}" if (self.out_dir / "previews" / f).is_file() else None

    def _room_card(self, room: str) -> str:
        scan = (self._scan(room) or ({}, ""))[0]
        sim, base = self._ds_stats(f"{room}_sim"), self._ds_stats(room)
        st = sim or base
        png = self._preview(f"{room}_sim", "contact.png")
        thumb = (
            f"<img src='{esc(png)}' alt='{esc(room)} simulation preview' loading='lazy'>" if png else "no preview yet"
        )
        chips = [_chip(scan.get("geometry_source", "unknown"))]
        if st is None:
            chips.append(_chip("not generated", "run"))
        elif st["valid"] is True:
            chips.append(_chip("physically valid: true", "ok"))
        else:
            chips.append(_chip("physically valid: false", "bad"))
        if sim and sim["generator"]:
            chips.append(_chip(sim["generator"]))
        if st:
            eps = "n/a" if st["episodes"] is None else st["episodes"]
            contacts = "n/a" if st["contacts"] is None else st["contacts"]
            stats = (
                f"<div class='stats'><div class='stat'><b>{esc(eps)}</b><span>episodes</span></div>"
                f"<div class='stat'><b>{esc(st['rate'])}</b><span>success</span></div>"
                f"<div class='stat'><b>{esc(contacts)}</b><span>human contacts</span></div></div>"
            )
        else:
            stats = (
                "<p class='muted'>No dataset yet. Run <code>python -m arrival demo</code> "
                "or <code>python -m arrival sim</code>.</p>"
            )
        acts = [f"<a class='btn primary' href='/rooms/{esc(room)}'>Open</a>"]
        if sim:
            acts.append(f"<a class='btn' href='/download/{esc(room)}_sim'>Sim zip</a>")
        if base:
            acts.append(f"<a class='btn' href='/download/{esc(room)}'>Placeholder zip</a>")
        return (
            f"<div class='card room'><div class='thumb'>{thumb}</div><div class='body'>"
            f"<h3><a href='/rooms/{esc(room)}'>{esc(room)}</a></h3><div class='chips'>{''.join(chips)}</div>"
            f"{stats}<div class='actions'>{''.join(acts)}</div></div></div>"
        )

    def _index(self):
        rooms = self._rooms()
        cards = (
            ("<div class='grid'>" + "".join(self._room_card(r) for r in rooms) + "</div>")
            if rooms
            else (
                "<div class='card empty'><b>No rooms yet</b>Put a scan under <code>scans/&lt;room&gt;/scan.json</code> (or <code>roomplan.json</code>) "
                "or upload one below.</div>"
            )
        )
        mb = self.server.max_upload_bytes / (1024 * 1024)
        upload = (
            "<form id='upload' method='post' action='/jobs' enctype='multipart/form-data'>"
            "<label class='drop'><b id='fname'>Choose or drop a scan file</b>"
            f"<div class='muted'>RoomPlan JSON or scan.json, up to {mb:g} MB in total</div>"
            "<input type='file' name='scan' accept='.json,application/json' required></label>"
            "<label class='drop sub'><b>Optional RGB-D frames</b>"
            "<div class='muted'>a .zip with <code>frames_manifest.json</code> at its root (CAP-2a). Held-out frames are compared with the room twin.</div><input type='file' name='frames' accept='.zip,application/zip'></label>"
            "<div class='upl'><button class='primary' type='submit'>Queue scan</button>"
            "<progress max='1' value='0' hidden></progress></div><div id='msg' role='status'></div></form>"
            "<p class='muted'>The file is stored under <code>out/jobs/&lt;id&gt;/</code> and queued. Nothing is "
            "compiled or generated in the request. Run <code>python -m arrival work</code> in another terminal to "
            "turn queued scans into sim datasets.</p>"
        )
        # PLACEHOLDER_NOTE is about the scripted_retarget dataset `demo` writes to out/datasets/<room>. Shown on every
        # visit, it told a page of physics-checked sim and upload datasets that its motion was "not physically valid"
        placeholder = any((st := self._ds_stats(r)) and st["valid"] is not True for r in rooms)
        note = f"<div class='warn'>{esc(PLACEHOLDER_NOTE)}</div>" if placeholder else ""
        body = (
            "<h1>Rooms to robot demos</h1><p class='muted'>Pre-scanned rooms become GR00T LeRobot v2 "
            "datasets, generated locally.</p>"
            "<ol class='flow'><li>Scan</li><li>Digital twin</li><li>MuJoCo sim</li><li>LeRobot dataset</li></ol>"
            + note
            + "<h2>Rooms</h2>"
            + cards
            + "<h2>Upload a scan</h2><div class='card'>"
            + upload
            + "</div>"
            + "<h2>Jobs</h2>"
            + self._retention_note()
            + "<div id='jobs'>"
            + self._jobs_table()
            + "</div>"
        )
        self._html(200, "Arrival", body)

    @staticmethod
    def _retention_note() -> str:
        try:
            hours = Limits.from_env().retention_hours
        except ValueError:
            hours = None
        keep = (
            f"Finished jobs and their datasets are deleted {hours:g} hours after they finish."
            if hours
            else "Finished jobs are kept until the person who uploaded them deletes them."
        )
        return (
            f"<p class='muted'>{keep} Download a dataset before then. The Cancel and Delete buttons "
            "appear only for jobs uploaded from this browser.</p>"
        )

    def _jobs(self) -> list[dict]:
        root = self.out_dir / "jobs"
        if not root.is_dir():
            return []
        jobs = [
            s
            for d in root.iterdir()
            if JOB_ID_RE.match(d.name) and isinstance(s := _read_json(d / "status.json"), dict) and s
        ]
        if self.server.public:  # a public page never lists other visitors' uploads
            owner = self._owner()
            jobs = [s for s in jobs if owner is not None and s.get("owner") == owner]
        return sorted(jobs, key=lambda s: (-s.get("created_unix", 0), s.get("id", "")))

    @staticmethod
    def _frames_chip(s: dict) -> str:
        fr = s.get("frames")
        if not isinstance(fr, dict):
            return ""
        measurement = s.get("realism") or {}
        state = measurement.get("state", "pending")
        label = {
            "pending": "comparison pending",
            "measured": "comparison ready",
            "invalid": "control failed",
            "not_measured": "not measured",
            "error": "comparison unavailable",
        }.get(state, "not measured")
        link = f" <a href='/jobs/{esc(s['id'])}/realism'>View comparison</a>" if measurement else ""
        return f"<div class='chips mt'>{_chip('RGB-D ' + str(fr.get('count', '?')) + ' frames, ' + label)}{link}</div>"

    @staticmethod
    def _objects_chip(s: dict) -> str:
        """Where the twin's objects came from: detected in the frames, written in the scan, or invented defaults."""
        ob = s.get("objects")
        if not isinstance(ob, dict):
            return ""
        counts = ob.get("counts") or {}
        text = "objects: " + (", ".join(f"{n} {k}" for k, n in counts.items() if n) or "none")
        if counts.get("default_synthetic"):
            text += " (invented, not scanned)"
        why = f"detection {ob.get('detection')}" + (f": {ob['reason']}" if ob.get("reason") else "")
        return f"<div class='chips mt'><span class='chip' title='{esc(why)}'>{esc(text)}</span></div>"

    def _jobs_table(self) -> str:
        rows = []
        for s in self._jobs():
            jid, st = s.get("id", "?"), s.get("status", "?")
            if st == "done" and s.get("zip"):
                result = f"<a class='btn' href='/jobs/{esc(jid)}/download'>dataset.zip</a>"
            elif st == "failed":
                result = f"<span class='chip bad' title='{esc(s.get('error'))}'>{esc(s.get('error') or 'failed')[:90]}</span>"
            else:
                result = ""
            eps = "" if s.get("attempts") is None else f"{esc(s.get('successes'))}/{esc(s.get('attempts'))}"
            mine = self._owner() is not None and s.get("owner") == self._owner()
            if mine:
                verb, label = ("delete", "Delete") if st in ("done", "failed", "cancelled") else ("cancel", "Cancel")
                result += f" <button type='button' data-job-action='{verb}' data-job-id='{esc(jid)}'>{label}</button>"
            kind = {"done": "ok", "failed": "bad", "running": "run"}.get(st, "")
            active = " data-active" if st in ("queued", "running") else ""
            rows.append(
                f"<tr{active}><td><a href='/jobs/{esc(jid)}'><code>{esc(jid)}</code></a></td>"
                f"<td>{esc(s.get('filename'))}{self._frames_chip(s)}{self._objects_chip(s)}</td><td>{_chip(st, kind)}</td><td>{_stage_bar(s)}</td>"
                f"<td><code>{esc(s.get('room_id') or '')}</code></td><td>{eps}</td><td>{result}</td></tr>"
            )
        if not rows:
            return (
                "<div class='card empty'><b>No uploads yet</b>Queued scans and their pipeline stages "
                "show up here.</div>"
            )
        return (
            "<div class='scroll'><table><tr><th>Job</th><th>Upload</th><th>Status</th><th>Progress</th><th>Room</th>"
            "<th>Episodes ok</th><th>Result</th></tr>" + "".join(rows) + "</table></div>"
        )

    def _fixtures_section(self, src: dict, twin: dict | None, scan_file) -> str:
        """The room page's fixture table, from the compiled twin when there is one, else the scan."""
        fixtures = src.get("fixtures", []) or []
        fx_rows = "".join(
            "<tr>"
            f"<td><code>{esc(f.get('id'))}</code></td>"
            f"<td>{esc(f.get('type', f.get('category', '')))}</td>"
            f"<td class='mono'>{esc(_fmt_list(f.get('size', '')))}</td>"
            f"<td class='mono'>{esc(_fmt_list(f.get('position', '')))}</td>"
            f"<td class='mono'>{esc(f.get('yaw', ''))}</td>"
            "</tr>"
            for f in fixtures
            if isinstance(f, dict)
        )
        fx_table = (
            f"<p class='muted'>Source: {'compiled twin.json' if twin else esc(scan_file) + ' (twin not compiled)'}</p>"
            "<div class='scroll'><table><tr><th>id</th><th>type</th><th>size (m)</th>"
            "<th>position (m)</th><th>yaw</th></tr>"
            + (fx_rows or "<tr><td colspan='5' class='muted'>No fixtures</td></tr>")
            + "</table></div>"
        )
        return fx_table

    def _tasks_section(self, room: str) -> str:
        """The room page's task summary: counts per family and instruction, or how to generate them."""
        tasks = _read_json(self.out_dir / "twins" / room / "tasks.json")
        if tasks and tasks.get("tasks"):
            grouped = Counter(
                (t.get("family", "?"), t.get("object_id", "?"), t.get("instruction", "")) for t in tasks["tasks"]
            )
            fam_counts = Counter(t.get("family", "?") for t in tasks["tasks"])
            summary = " · ".join(f"<code>{esc(f)}</code>: {n}" for f, n in sorted(fam_counts.items()))
            t_rows = "".join(
                f"<tr><td><code>{esc(fam)}</code></td><td><code>{esc(obj)}</code></td>"
                f"<td>{esc(instr)}</td><td>{n}</td></tr>"
                for (fam, obj, instr), n in sorted(grouped.items())
            )
            tasks_html = (
                f"<p>{summary}</p><div class='scroll'><table><tr><th>family</th><th>object</th>"
                "<th>instruction</th><th>count</th></tr>" + t_rows + "</table></div>"
            )
        else:
            tasks_html = (
                "<p class='muted'>Families <code>counter_to_sink</code> and "
                "<code>counter_to_counter</code>. Not generated, run "
                "<code>python -m arrival demo</code>.</p>"
            )
        return tasks_html

    def _report_section(self, room: str) -> str:
        """The room page's speed and cost report for the placeholder dataset, if one was written."""
        report = _read_json(self.out_dir / "reports" / f"{room}.json")
        if report:
            rep_rows = "".join(
                f"<tr><th>{esc(k)}</th><td><code>{esc(report.get(k))}</code></td></tr>"
                for k in ("attempts", "successes", "dropped", "seconds", "demos_per_hour")
            )
            dollars = report.get("dollars") or {}
            d_rows = "".join(
                f"<tr><th>{esc(k)}</th><td><code>{esc(v.get('usd') if isinstance(v, dict) else v)}</code></td>"
                f"<td>{'measured' if isinstance(v, dict) and v.get('measured') else 'not measured'}</td></tr>"
                for k, v in dollars.items()
            )
            report_html = (
                f"<div class='scroll'><table class='kv'>{rep_rows}</table></div>"
                "<h2>Cost (USD)</h2><div class='scroll'><table><tr><th>item</th><th>usd</th>"
                f"<th>status</th></tr>{d_rows}</table></div>"
            )
        else:
            report_html = f"<p class='muted'>No report at <code>out/reports/{esc(room)}.json</code> yet.</p>"
        return report_html

    def _sim_section(self, room: str, sim_dir: Path) -> str:
        """The room page's MuJoCo dataset block: download link and its labels, or how to make one."""
        sim_info = _read_json(sim_dir / "meta" / "info.json") if sim_dir.is_dir() else None
        if sim_dir.is_dir():
            si = sim_info if isinstance(sim_info, dict) else {}
            fams = si.get("arrival_families") or si.get("families") or []
            sim_html = (
                f"<p><a class='btn primary' href='/download/{esc(room)}_sim'>Download {esc(room)}_sim.zip</a></p>"
                "<div class='scroll'><table class='kv'>"
                f"<tr><td>arrival_generator</td><td class='mono'>{esc(si.get('arrival_generator', 'unknown'))}</td></tr>"
                f"<tr><td>arrival_physically_valid</td><td class='mono'>{esc(si.get('arrival_physically_valid', 'unknown'))}</td></tr>"
                f"<tr><td>episodes</td><td class='mono'>{esc(si.get('total_episodes', 'unknown'))}</td></tr>"
                f"<tr><td>families</td><td class='mono'>{esc(', '.join(map(str, fams)) if isinstance(fams, list) else fams)}</td></tr>"
                "</table></div>"
            )
        else:
            sim_html = (
                f"<p class='muted'>No simulation dataset at <code>out/datasets/{esc(room)}_sim</code>. "
                "Run <code>python -m arrival sim</code>.</p>"
            )
        return sim_html

    def _room(self, room: str):
        """GET /rooms/<room>: the room's page (fixtures, tasks, both datasets, previews and report)."""
        found = self._scan(room)
        if found is None:
            return self._not_found(f"No room {room}")
        scan, scan_file = found
        twin = _read_json(self.out_dir / "twins" / room / "twin.json")
        src = twin if twin else scan
        fx_table = self._fixtures_section(src, twin, scan_file)
        tasks_html = self._tasks_section(room)
        info = _read_json(self.out_dir / "datasets" / room / "meta" / "info.json") or {}
        valid = info.get("arrival_physically_valid", False)
        valid_text = "true" if valid is True else "false"
        info_bits = ""
        if info:
            info_bits = "".join(
                f"<tr><th>{esc(k)}</th><td><code>{esc(info.get(k))}</code></td></tr>"
                for k in (
                    "arrival_generator",
                    "arrival_appearance",
                    "arrival_video_policy",
                    "total_episodes",
                    "total_frames",
                    "arrival_successes",
                    "arrival_attempts",
                    "robot_type",
                    "fps",
                )
                if k in info
            )
            info_bits = f"<div class='scroll'><table class='kv'>{info_bits}</table></div>"

        report_html = self._report_section(room)
        ds_exists = (self.out_dir / "datasets" / room).is_dir()
        dl = (
            f"<p><a class='btn primary' href='/download/{esc(room)}'>Download {esc(room)}.zip</a></p>"
            if ds_exists
            else "<p class='muted'>Dataset not generated yet.</p>"
        )

        sim_dir = self.out_dir / "datasets" / f"{room}_sim"
        sim_html = self._sim_section(room, sim_dir)
        png = self._preview(f"{room}_sim", "contact.png")
        mp4 = self._preview(f"{room}_sim", "overview.mp4")
        previews = ""
        if png:
            previews += (
                f"<figure><img src='{esc(png)}' alt='contact sheet' loading='lazy'>"
                "<figcaption>Contact sheet from the simulation</figcaption></figure>"
            )
        if mp4:
            previews += (
                f"<figure><video src='{esc(mp4)}' controls muted loop playsinline preload='metadata'>"
                "</video><figcaption>Overview render</figcaption></figure>"
            )
        preview_html = (
            f"<div class='media'>{previews}</div>"
            if previews
            else ("<p class='muted'>No previews yet. They are written by <code>python -m arrival sim</code>.</p>")
        )
        stats = self._ds_stats(f"{room}_sim") or self._ds_stats(room)
        if stats:
            ok_ = stats["valid"] is True
            headline = (
                "<div class='stats'>"
                f"<div class='stat'><b>{esc('n/a' if stats['episodes'] is None else stats['episodes'])}</b><span>episodes</span></div>"
                f"<div class='stat'><b>{esc(stats['rate'])}</b><span>success rate</span></div>"
                f"<div class='stat'><b>{esc('n/a' if stats['contacts'] is None else stats['contacts'])}</b><span>human contacts</span></div></div>"
                f"<div class='chips mt'>{_chip('physically valid: ' + str(ok_).lower(), 'ok' if ok_ else 'bad')}</div>"
            )
        else:
            headline = ""
        # the box is about the scripted placeholder in out/datasets/<room>. A room whose only dataset is the
        # physics-checked sim one has no placeholder, and the box told it not to fine-tune on its data right above
        # a headline that says physically valid: true
        placeholder_warn = (
            ""
            if sim_dir.is_dir() and not ds_exists
            else (
                f"<div class='warn'><strong class='mono'>arrival_physically_valid: {valid_text}</strong>"
                f"<br>{esc(PLACEHOLDER_NOTE)} Do not fine-tune on it while this flag is false.</div>"
            )
        )

        body = (
            f"<p class='crumb'><a href='/'>← All rooms</a></p><h1>{esc(room)}</h1>"
            f"<p class='muted'>geometry_source: <code>{esc(scan.get('geometry_source', 'unknown'))}</code></p>"
            + placeholder_warn
            + headline
            + "<h2>Previews</h2>"
            + preview_html
            + "<h2>Fixtures</h2>"
            + fx_table
            + "<h2>Tasks</h2>"
            + tasks_html
            + "<h2>Dataset</h2>"
            + dl
            + info_bits
            + "<h2>Simulation dataset (MuJoCo)</h2>"
            + sim_html
            + "<h2>Report</h2>"
            + report_html
        )
        self._html(200, f"{room} · Arrival", body)

    def _preview_file(self, name: str):
        """Serve one png/mp4/gif from out/previews, with Range support so browsers can seek video."""
        f = self.out_dir / "previews" / name
        ctype = PREVIEW_TYPES.get(f.suffix.lower())
        if not PREVIEW_RE.match(name) or ctype is None or not f.is_file() or not self._room_asset_allowed(name):
            return self._not_found("No such preview")
        size = f.stat().st_size
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        start, end, code = 0, size - 1, 200
        if m and (m.group(1) or m.group(2)):
            try:
                if m.group(1):
                    start, end = int(m.group(1)), min(int(m.group(2) or size - 1), size - 1)
                else:
                    start = max(size - int(m.group(2)), 0)
            except ValueError:  # ignore ranges exceeding Python's integer conversion limit
                start, end = 0, size - 1
            else:
                if start > end or start >= size:
                    return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
                code = 206
        extra = {"Accept-Ranges": "bytes"}
        if code == 206:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self._start(code, ctype, end - start + 1, extra, cache="private, max-age=30")
        self.end_headers()
        if self.command != "HEAD":
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0 and (chunk := fh.read(min(left, READ_CHUNK))):
                    self.wfile.write(chunk)
                    left -= len(chunk)

    def _download(self, room: str):
        ds = self.out_dir / "datasets" / room
        if not ds.is_dir() or not self._room_asset_allowed(room):
            return self._not_found(f"No dataset for {room}")
        if not self.server.heavy_slots.acquire(blocking=False):  # zipping a dataset is the other costly request
            return self._error(503, "server is busy, retry shortly", {"Retry-After": "5"})
        try:
            with tempfile.TemporaryFile() as tmp:  # spooled to disk, then streamed
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in sorted(ds.rglob("*")):
                        if p.is_file():
                            zf.write(p, Path(room) / p.relative_to(ds))
                size = tmp.tell()
                tmp.seek(0)
                self._send_file(
                    tmp,
                    size,
                    "application/zip",
                    {"Content-Disposition": f"attachment; filename={hd.safe_filename(room)}.zip"},
                )
        finally:
            self.server.heavy_slots.release()

    def _public_status(self, status: dict) -> dict:
        """The status as the API returns it: no owner hash, and in public mode no server paths."""
        out = {k: v for k, v in status.items() if k != "owner"}
        if self.server.public:
            for k in ("twin_dir", "dataset_dir", "zip"):
                out.pop(k, None)
        return out

    def _room_asset_allowed(self, name: str) -> bool:
        """Uploaded sim datasets and previews inherit the job's public-mode owner check."""
        if not self.server.public:
            return True
        match = UPLOADED_ASSET_RE.search(name)
        if match is None:
            return True
        status = read_job(self.out_dir / "jobs", match.group(1))
        owner = self._owner()
        return (
            status is not None
            and owner is not None
            and status.get("owner") == owner
            and status.get("room_id") == name[: match.end(1)]
        )

    def _job_for_request(self, job_id: str) -> dict | None:
        status = read_job(self.out_dir / "jobs", job_id)
        if status is None:
            return None
        if self.server.public:
            owner = self._owner()
            if owner is None or status.get("owner") != owner:
                return None
        return status

    def _job(self, job_id: str):
        status = self._job_for_request(job_id)
        if status is None:
            return self._json(404, {"error": "job not found"})
        self._json(200, self._public_status(status))

    def _delete_job(self, job_id: str):
        """Cancel (queued or running) or delete (finished) a job, for the browser that uploaded it only. Any other
        caller, and an unknown id, gets the same 404."""
        from arrival.jobs.retention import FINISHED, JobBusy, delete_job
        from arrival.jobs.worker import cancel_job

        status = read_job(self.out_dir / "jobs", job_id)
        owner = self._owner()
        if status is None or owner is None or status.get("owner") != owner:
            return self._json(404, {"error": "job not found"})
        if status.get("status") in FINISHED:
            try:
                res = delete_job(self.out_dir, job_id)
            except JobBusy as e:
                return self._json(409, {"error": str(e)})
            if res is None:
                return self._json(404, {"error": "job not found"})
            return self._json(200, {"id": job_id, "result": "deleted"})
        res = cancel_job(self.out_dir / "jobs", job_id)
        if res is None:
            return self._json(404, {"error": "job not found"})
        if res == "cancelling":
            return self._json(202, {"id": job_id, "result": "cancelling"})
        if res == "cancelled":
            return self._json(200, {"id": job_id, "result": "cancelled"})
        return self._json(409, {"error": f"job is {res}; try again"})

    def _job_realism(self, job_id: str, artifact: str):
        status = read_job(self.out_dir / "jobs", job_id)
        if status is None or (self.server.public and (self._owner() is None or self._owner() != status.get("owner"))):
            return self._not_found()
        measurement = status.get("realism") or {
            "state": "pending",
            "metrics": None,
            "preview": False,
            "reason": "The worker has not compared these frames yet.",
        }
        if artifact == "report":
            return self._json(200, measurement)
        if artifact == "preview":
            path = self.out_dir / "jobs" / job_id / "realism" / "comparison.png"
            if not measurement.get("preview") or not path.is_file():
                return self._not_found()
            with path.open("rb") as stream:
                return self._send_file(stream, path.stat().st_size, "image/png", {"Cache-Control": "no-store"})
        metrics = measurement.get("metrics")
        body = (
            "<h1>Room comparison</h1><p><a href='/'>Back to jobs</a></p>"
            f"<div class='card'><h2>{esc(measurement['state'].replace('_', ' '))}</h2>"
        )
        if measurement.get("reason"):
            body += f"<p>{esc(measurement['reason'])}</p>"
        if metrics:
            body += (
                "<dl>"
                f"<dt>Depth error (geometry)</dt><dd>{metrics['depth_l1_m']:.4f} m</dd>"
                f"<dt>RGB error (appearance)</dt><dd>{metrics['rgb_mae']:.4f}</dd>"
                f"<dt>Held-out frames</dt><dd>{metrics['frames']}</dd></dl>"
            )
        body += (
            "<p class='muted'>Depth compares room geometry. RGB also reflects differences in lighting and "
            "surface appearance. The camera-offset control checks sensitivity; it does not verify calibration "
            "or that the captures belong to this room.</p>"
        )
        if measurement.get("preview"):
            body += (
                "<div class='media'><figure>"
                f"<img src='/jobs/{esc(job_id)}/realism/preview' "
                "alt='Uploaded held-out views beside the simulated room and depth maps'>"
                "<figcaption>Reference RGB, twin RGB, reference depth, twin depth.</figcaption></figure></div>"
            )
        body += f"<p><a href='/jobs/{esc(job_id)}/realism/report'>Comparison JSON</a></p></div>"
        self._html(200, "Room comparison", body)

    def _job_download(self, job_id: str):
        status = self._job_for_request(job_id)
        if status is None:
            return self._json(404, {"error": "job not found"})
        zip_path = self.out_dir / "jobs" / job_id / "dataset.zip"
        if status.get("status") != "done" or not zip_path.is_file():
            return self._json(404, {"error": "dataset not ready", "status": status.get("status")})
        room = status.get("room_id") or job_id
        with open(zip_path, "rb") as f:
            self._send_file(
                f,
                zip_path.stat().st_size,
                "application/zip",
                {"Content-Disposition": f"attachment; filename={hd.safe_filename(room, job_id)}_sim.zip"},
            )

    def _drain(self, left: int):
        """Read and discard a small unread request body, so closing the socket does not reset the connection
        before the client has read our answer. A large one is not read: the connection is just closed."""
        self.close_connection = True
        if 0 < left <= DRAIN_MAX:
            try:
                while left > 0 and (chunk := self.rfile.read(min(left, READ_CHUNK))):
                    left -= len(chunk)
            except OSError:
                pass

    def _active_jobs(self) -> int:
        root = self.out_dir / "jobs"
        if not root.is_dir():
            return 0
        n = 0
        for d in root.iterdir():
            if JOB_ID_RE.match(d.name):
                st = _read_json(d / "status.json")
                n += isinstance(st, dict) and st.get("status") in ("queued", "running")
        return n

    def _queue_full(self) -> bool:
        return self.server.max_queued > 0 and self._active_jobs() >= self.server.max_queued

    def _free_disk(self) -> int:
        p = self.out_dir
        while not p.exists() and p != p.parent:
            p = p.parent
        return shutil.disk_usage(p).free

    def _post_job(self):
        """POST /jobs: accept a scan upload (and an optional RGB-D frames zip) and queue a job. Every cheap refusal
        (rate limit, size, queue cap, content type, disk space, busy server) happens before the body is read; the
        body is then streamed to a temp file, its parts are sniffed, and the job is created under the queue lock."""
        srv = self.server
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 0:
                raise ValueError
        except ValueError:
            self.close_connection = True
            return self._json(411, {"error": "Content-Length required"})

        def refuse(code: int, error: str, extra: dict | None = None, **more):
            self._drain(length)  # the body is still unread
            self._json(code, {"error": error, **more}, extra)

        try:
            limits = Limits.from_env()
        except LimitError as e:  # a bad ARRIVAL_* variable is the operator's mistake, not the client's
            return refuse(500, f"server limits misconfigured: {e}")
        max_bytes = (
            min(srv.max_upload_bytes, limits.max_upload_bytes) if limits.max_upload_bytes else srv.max_upload_bytes
        )

        allowed, wait = srv.limiter.check(self._client())
        if not allowed:
            return refuse(429, "too many uploads, retry later", {"Retry-After": str(wait)})
        if not check_upload_size(length, max_bytes):
            return refuse(413, "upload too large", max_bytes=max_bytes)
        if limits.max_queue and count_active(self.out_dir / "jobs") >= limits.max_queue:  # before reading the body
            return refuse(429, "queue is full, try again later", max_queue=limits.max_queue)
        ctype = self.headers.get("Content-Type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            return refuse(400, "expected multipart/form-data")
        m = BOUNDARY_RE.search(ctype)
        if not m or len(m.group(1)) > MAX_BOUNDARY:
            return refuse(400, "multipart boundary missing or too long")
        if self._queue_full():
            return refuse(503, "queue is full, try again when queued jobs finish", {"Retry-After": "60"})
        if length and self._free_disk() < 2 * length + DISK_MARGIN:
            return refuse(507, "not enough free disk space for this upload")
        if not srv.heavy_slots.acquire(blocking=False):
            return refuse(503, "server is busy with other uploads, retry shortly", {"Retry-After": "5"})
        jobs_root = self.out_dir / "jobs"
        incoming = None
        try:
            with tempfile.TemporaryFile() as tmp:
                left, deadline = length, time.monotonic() + UPLOAD_DEADLINE
                while left > 0:  # stream to disk; never hold the whole body in memory
                    if time.monotonic() > deadline:
                        self.close_connection = True
                        return self._json(408, {"error": "upload took too long"})
                    chunk = self.rfile.read(min(left, READ_CHUNK))
                    if not chunk:
                        break
                    tmp.write(chunk)
                    left -= len(chunk)
                if left:
                    self.close_connection = True
                    return self._json(400, {"error": "truncated body", "expected": length, "got": length - left})
                if length == 0:
                    return self._json(400, {"error": "empty body"})
                tmp.flush()
                with mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                    boundary = m.group(1).encode("latin-1")
                    try:
                        offset, size, filename = find_part(buf, boundary, "scan")
                        frames_part = find_part(buf, boundary, "frames", optional=True)
                        if frames_part and frames_part[1] == 0 and not frames_part[2]:
                            frames_part = (
                                None  # a browser sends the optional input left empty as a nameless, empty part
                            )
                    except UploadError as e:
                        return self._json(400, {"error": str(e)})
                    head = bytes(buf[offset : offset + min(size, 64)])
                    fhead = bytes(buf[frames_part[0] : frames_part[0] + min(frames_part[1], 4)]) if frames_part else b""
                if not filename:
                    return self._json(400, {"error": "file field 'scan' has no filename"})
                if size <= 0:
                    return self._json(400, {"error": "uploaded file is empty"})
                name = hd.safe_filename(filename)
                if not hd.has_extension(name, hd.SCAN_EXTENSIONS):
                    return self._json(415, {"error": "the scan must be a .json file (RoomPlan export or scan.json)"})
                if not hd.looks_like_json_object(head):
                    return self._json(415, {"error": "the scan file is not a JSON object"})
                frames_summary, adopt = None, None
                if frames_part:
                    f_off, f_size, f_name = frames_part
                    if f_size <= 0:
                        return self._json(400, {"error": "frames file is empty", "field": "frames"})
                    if not hd.has_extension(hd.safe_filename(f_name), hd.FRAMES_EXTENSIONS) or not hd.looks_like_zip(
                        fhead
                    ):
                        return self._json(415, {"error": "frames must be a .zip of a CAP-2a bundle", "field": "frames"})
                    jobs_root.mkdir(parents=True, exist_ok=True)
                    incoming = jobs_root / f".incoming_{secrets.token_hex(8)}"
                    try:
                        frames_mod.extract(tmp, f_off, f_size, incoming)
                        frames_summary = frames_mod.validate(incoming)
                    except frames_mod.FramesError as e:
                        shutil.rmtree(
                            incoming, ignore_errors=True
                        )  # before answering, so no scratch outlives the reply
                        code = 413 if e.code == "too_large" else 422 if e.code not in frames_mod.ARCHIVE_CODES else 400
                        return self._json(code, {"error": str(e), "code": e.code, "field": "frames"})
                    adopt = {"frames": incoming}
                token = hd.cookie_value(self.headers.get("Cookie"))
                new_cookie = token is None
                token = token or secrets.token_hex(16)
                extra_status = {"owner": hd.owner_of(token)}
                if frames_summary:
                    extra_status["frames"] = frames_summary
                with srv.post_lock:  # the cap is checked again under the lock, so a burst cannot overshoot it
                    if self._queue_full():
                        if incoming is not None:
                            shutil.rmtree(incoming, ignore_errors=True)
                        return self._json(
                            503, {"error": "queue is full, try again when queued jobs finish"}, {"Retry-After": "60"}
                        )
                    status = create_job(
                        jobs_root, name, src=tmp, offset=offset, length=size, extra_status=extra_status, adopt=adopt
                    )
        finally:
            srv.heavy_slots.release()
            if incoming is not None and incoming.exists():
                shutil.rmtree(incoming, ignore_errors=True)
        extra = {}
        if new_cookie:
            extra["Set-Cookie"] = f"{hd.CLIENT_COOKIE}={token}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax" + (
                "; Secure" if self.https else ""
            )
        body = {"status": "queued", "id": status["id"], "status_url": f"/jobs/{status['id']}"}
        if frames_summary:
            body["frames"] = frames_summary
        self._json(202, body, extra)


def make_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    out_dir: Path = Path("out"),
    scans_dir: Path = Path("scans"),
    *,
    max_upload_mb: float = MAX_UPLOAD_BYTES / (1024 * 1024),
    rate_limit: int = 10,
    max_queued: int = 50,
    public: bool = False,
    behind_proxy: bool = False,
    max_concurrent_uploads: int = 4,
) -> ThreadingHTTPServer:
    """``rate_limit`` is uploads per minute per client (0 = off); ``max_queued`` caps queued + running jobs
    (0 = unlimited); ``public`` lists only the caller's own jobs; ``behind_proxy`` trusts X-Forwarded-Proto and
    X-Forwarded-For from a reverse proxy in front (never set it when clients can reach the port directly)."""
    server = ArrivalServer((host, port), Handler)
    server.out_dir = Path(out_dir)
    server.scans_dir = Path(scans_dir)
    server.max_upload_bytes = int(max_upload_mb * 1024 * 1024)
    server.limiter = hd.RateLimiter(rate_limit)
    server.max_queued = int(max_queued)
    server.public = bool(public)
    server.behind_proxy = bool(behind_proxy)
    server.post_lock = threading.Lock()
    server.heavy_slots = threading.BoundedSemaphore(max(1, int(max_concurrent_uploads)))
    return server


def lan_ip() -> str | None:
    """This machine's address on the local network (no packet is sent), or None when offline."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1: picks the outbound interface without traffic
            return s.getsockname()[0]
        except OSError:
            return None


def serve(
    port: int = 8000, out_dir: Path = Path("out"), scans_dir: Path = Path("scans"), host: str = "127.0.0.1", **limits
) -> None:
    """Serve the page on host:port until Ctrl+C, printing the URL (and the LAN address for the capture app when bound to
    all interfaces) and a warning whenever it is reachable beyond loopback."""
    server = make_server(host, port, out_dir, scans_dir, **limits)
    bound, real_port = server.server_address[:2]
    print(f"Arrival serving on http://{bound}:{real_port}/  (Ctrl+C to stop)", flush=True)
    if bound == "0.0.0.0" and (ip := lan_ip()):
        print(f"Capture app server URL: http://{ip}:{real_port}", flush=True)
    if bound not in LOOPBACK:
        print(
            "No authentication: only expose this beyond a network you trust behind a TLS and auth proxy "
            "(see docs/REFERENCE.md, Web hardening).",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
