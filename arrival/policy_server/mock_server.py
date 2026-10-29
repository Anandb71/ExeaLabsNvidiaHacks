"""Offline stand-in for a GR00T policy server (stdlib only), for tests and wiring checks.

    python -m arrival.policy_server.mock_server --port 8000 --mode hold

Modes: hold (the action chunk repeats the current state), noise (state plus seeded gaussian noise),
fail_after_n (hold for N requests, then HTTP 500). It does not look at the images or the instruction, so a run against
it proves the plumbing, not any policy. --port 0 binds a free port and prints it.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from arrival.policy_server.protocol import ProtocolError, decode_request, encode_response

MODES = ("hold", "noise", "fail_after_n")


class MockPolicyServer(ThreadingHTTPServer):
    """An offline stand-in for a GR00T policy server on the protocol's /act and /health routes. `mode` picks the replies
    (hold the pose, hold with noise, or fail_after_n: answer `fail_after` requests and then return HTTP 500), `chunk`
    the steps per reply. It never looks at the images or instruction."""

    daemon_threads = True

    def __init__(
        self,
        port: int = 0,
        mode: str = "hold",
        chunk: int = 8,
        seed: int = 0,
        noise_std: float = 0.01,
        fail_after: int = 0,
        host: str = "127.0.0.1",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
            # 0 made every /act a 500 and a negative count dropped the connection with no reply
            raise ValueError(f"chunk must be a whole number of steps, at least 1, got {chunk!r}")
        super().__init__((host, port), _Handler)
        self.mode, self.chunk, self.noise_std, self.fail_after = mode, chunk, noise_std, fail_after
        self.rng = np.random.default_rng(seed)
        self.requests = 0  # /act requests seen, failed ones included
        self.lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://{self.server_address[0]}:{self.server_address[1]}"

    def start(self) -> MockPolicyServer:
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def act(self, state: np.ndarray) -> np.ndarray | None:
        """Chunk [T,8] for this state, or None when the server is set to fail."""
        with self.lock:
            self.requests += 1
            if self.mode == "fail_after_n" and self.requests > self.fail_after:
                return None
            chunk = np.tile(state, (self.chunk, 1))
            if self.mode == "noise":
                chunk = chunk + self.rng.normal(0.0, self.noise_std, chunk.shape)
        return chunk


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "server": "mock", "mode": self.server.mode})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/act":
            return self._send(404, {"error": "not found"})
        try:
            req = decode_request(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
        except (ProtocolError, ValueError) as e:
            return self._send(400, {"error": str(e)})
        chunk = self.server.act(req["state"])
        if chunk is None:
            return self._send(500, {"error": "mock server set to fail"})
        try:
            body = encode_response(chunk)
        except ProtocolError as e:  # a bad chunk is a server fault: answer 500 rather than dropping the connection
            return self._send(500, {"error": str(e)})
        self._send(200, body)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8000, help="0 picks a free port")
    p.add_argument("--mode", choices=MODES, default="hold")
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fail-after", type=int, default=0, help="requests answered before fail_after_n starts failing")
    a = p.parse_args(argv)
    if a.chunk < 1:
        p.error("--chunk must be at least 1 step")
    srv = MockPolicyServer(a.port, a.mode, a.chunk, a.seed, fail_after=a.fail_after)
    print(f"mock policy server ({a.mode}) at {srv.url}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
