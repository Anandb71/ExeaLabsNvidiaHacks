"""L4 helpers for the web server: safe filenames, upload sniffing, a per-client rate limiter, log redaction,
and the security headers. Standard library only; nothing here touches the pipeline."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import threading
import time
import unicodedata
from collections import deque

JOB_ID_RE = re.compile(r"^job_[0-9a-f]{8,32}$")
CLIENT_COOKIE = "arrival_client"
CLIENT_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
SCAN_EXTENSIONS = (".json",)
FRAMES_EXTENSIONS = (".zip",)
ZIP_MAGIC = b"PK\x03\x04"
FILENAME_KEEP = re.compile(r"[^A-Za-z0-9._() -]+")
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)
BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
KEY_VALUE = re.compile(r"(?i)\b([A-Za-z0-9_]*(?:api[_-]?key|token|secret|password)[A-Za-z0-9_]*)\s*[=:]\s*[^\s,;&\"']+")


def safe_filename(name: str | None, default: str = "upload.bin", max_len: int = 80) -> str:
    """A display and bookkeeping name from a client-supplied one: no directories (either slash), no control
    characters, only ``A-Za-z0-9._() -``, no leading dots, at most ``max_len`` characters with the extension kept.
    A name with no letter or digit left before its extension (``Кухня.json``, ``厨房.json``) keeps the extension on
    the default's stem (``upload.json``). The name is never used to build a path (the upload is always stored as
    ``upload.bin``)."""
    text = unicodedata.normalize("NFKC", name or "").replace("\\", "/").split("/")[-1]
    text = FILENAME_KEEP.sub("_", "".join(c for c in text if c.isprintable()))
    stem, dot, ext = text.rpartition(".")
    if dot and re.fullmatch(r"[A-Za-z0-9]{1,8}", ext) and not re.search(r"[A-Za-z0-9]", stem):
        return f"{default.partition('.')[0]}.{ext}"  # the strip below would leave only "json", which has no extension
    text = re.sub(r"_{2,}", "_", text).strip(" ._") if text.strip(" .") else ""
    if not text or not re.search(r"[A-Za-z0-9]", text):
        return default
    if len(text) > max_len:
        stem, dot, ext = text.rpartition(".")
        if dot and 0 < len(ext) <= 8:
            text = stem[: max_len - len(ext) - 1] + "." + ext
        else:
            text = text[:max_len]
    return text


def has_extension(name: str, allowed: tuple[str, ...]) -> bool:
    return name.lower().endswith(allowed)


def looks_like_json_object(head: bytes) -> bool:
    """True when the first non-blank byte (after an optional UTF-8 BOM) is ``{``. A cheap sniff, not a parse."""
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:]
    return head.lstrip(b" \t\r\n")[:1] == b"{"


def looks_like_zip(head: bytes) -> bool:
    return head.startswith(ZIP_MAGIC)


def owner_of(token: str | None) -> str | None:
    """The value stored as ``owner`` in a job's status: a hash of the client cookie, never the cookie itself."""
    if not token or not CLIENT_TOKEN_RE.match(token):
        return None
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def cookie_value(header: str | None, name: str = CLIENT_COOKIE) -> str | None:
    """The client token from a Cookie header, only if it has the token's exact shape."""
    for part in (header or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == name and CLIENT_TOKEN_RE.match(v):
            return v
    return None


def client_ip(peer: str, forwarded_for: str | None, behind_proxy: bool) -> str:
    """The rate-limit key. X-Forwarded-For is honoured only behind a trusted proxy, and only its last entry (the
    one the proxy itself appended); anything that is not an IP address falls back to the socket peer."""
    if behind_proxy and forwarded_for:
        last = forwarded_for.split(",")[-1].strip()
        try:
            return str(ipaddress.ip_address(last))
        except ValueError:
            pass
    return peer


class RateLimiter:
    """At most ``limit`` hits per ``window`` seconds per key (sliding window). ``limit`` 0 turns it off."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic, max_keys: int = 4096):
        self.limit, self.window, self.clock, self.max_keys = int(limit), float(window), clock, max_keys
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """(allowed, seconds until a retry could succeed). Records the hit only when allowed."""
        if self.limit <= 0:
            return True, 0
        now = self.clock()
        with self._lock:
            if len(self._hits) >= self.max_keys:  # drop keys whose window has passed so the table stays bounded
                for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
                    del self._hits[k]
                if len(self._hits) >= self.max_keys:
                    self._hits.clear()
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, int(self.window - (now - q[0]) + 0.999))
            q.append(now)
            return True, 0


def redact(text: str, environ=None) -> str:
    """Mask secrets in text before it reaches a log: the values of secret-named environment variables, bearer and
    basic credentials, and ``api_key=...`` style pairs."""
    environ = os.environ if environ is None else environ
    out = str(text)
    for name, value in environ.items():
        if SECRET_NAME.search(name) and len(value) >= 4:
            out = out.replace(value, "[redacted]")
    out = BEARER.sub(lambda m: f"{m.group(1)} [redacted]", out)
    return KEY_VALUE.sub(lambda m: f"{m.group(1)}=[redacted]", out)


def csp_hash(text: str) -> str:
    """The CSP 'sha256-...' source for an inline script or style."""
    import base64

    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"


def security_headers(csp: str, https: bool = False) -> dict[str, str]:
    """The headers every response carries: the given CSP, no sniffing or framing, no referrer, no device permissions,
    same-origin isolation, and HSTS when served over HTTPS."""
    h = {
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    if https:
        h["Strict-Transport-Security"] = "max-age=31536000"
    return h
