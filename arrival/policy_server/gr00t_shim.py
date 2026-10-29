"""The /act + /health contract of arrival.policy_server.protocol wrapped around Isaac-GR00T's Gr00tPolicy.

Run against real Isaac-GR00T (commit dac54b3, N1.7-3B, torch 2.9 / CUDA 12.8) on a 96 GB NVIDIA Blackwell GPU, with a
checkpoint from a 100-step fine-tune on hero_kitchen_sim: the imports, the `Gr00tPolicy` constructor and `get_action`
worked as written, and `/act` returned [16,7] and [16,1] chunks. On 13 frames from the training data the first step
sat 0.08 rad (mean) from the current joint state while its magnitude was about 0.73, so relative arm actions come back
already absolute, as the client assumes. Small sample, one checkpoint, fixed-base 8-float schema only; the mobile
(11-float) schema has not been run. See docs/ref/POL-3.md and docs/ref/GPU-RUN.md.

    python -m arrival.policy_server.gr00t_shim --model-path <checkpoint dir> --port 8000 \
        --modality-config-path configs/gr00t/arrival_panda_config.py

then `export GR00T_POLICY_URL=http://<host>:8000` on the machine running `python -m arrival eval --policy vla`.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from arrival.policy_server.protocol import ARM_DOF, TASK_KEY, VIDEO_KEYS, ProtocolError, decode_request, encode_response


def load_policy(model_path: str, modality_config_path: str, device: str, embodiment_tag: str):
    """Import gr00t lazily; exit with a message when it is not installed."""
    try:
        import importlib.util

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
    except ImportError as e:
        sys.exit(
            f"gr00t_shim: Isaac-GR00T is not installed here ({e}). Install it on the GPU box "
            "(https://github.com/NVIDIA/Isaac-GR00T) and run this module there."
        )
    # importing the config registers the Arrival Panda modality config for NEW_EMBODIMENT
    spec = importlib.util.spec_from_file_location("arrival_panda_config", modality_config_path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return Gr00tPolicy(embodiment_tag=EmbodimentTag(embodiment_tag), model_path=model_path, device=device)


def to_observation(req: dict, batch: int = 1) -> dict:
    """Decoded request -> Gr00tPolicy observation (one time step). `batch` > 1 repeats the same observation so the
    policy draws that many independent samples (it starts every chunk from fresh random noise)."""
    state = req["state"]

    def rep(x: np.ndarray) -> np.ndarray:
        return np.repeat(x, batch, axis=0) if batch > 1 else x

    st = {
        "single_arm": rep(state[None, None, :ARM_DOF].astype(np.float32)),
        "gripper": rep(state[None, None, ARM_DOF : ARM_DOF + 1].astype(np.float32)),
    }
    if state.shape[0] > ARM_DOF + 1:  # mobile schema
        st["base"] = rep(state[None, None, ARM_DOF + 1 :].astype(np.float32))
    return {
        "video": {k: rep(req["video"][k][None, None]) for k in VIDEO_KEYS},
        "state": st,
        "language": {TASK_KEY: [[req["instruction"]] for _ in range(batch)]},
    }


def to_chunk(action: dict, reduce: str = "mean") -> np.ndarray:
    """Gr00tPolicy action dict ({key: [B,T,D]}) -> chunk [T,8], or [T,11] when the action has a base key. A batch of
    B samples is reduced to one chunk by the mean or the median over B (B = 1 is a plain copy)."""
    if reduce not in ("mean", "median"):
        raise ValueError(f"unknown reduce {reduce!r}; choose mean or median")
    keys = ("single_arm", "gripper") + (("base",) if "base" in action else ())
    fn = np.median if reduce == "median" else np.mean
    return np.concatenate([fn(np.asarray(action[k], dtype=np.float64), axis=0) for k in keys], axis=1)


def make_handler(policy, samples: int = 1, reduce: str = "mean"):
    """The HTTP handler class for the shim: GET /health, and POST /act, which decodes the request, asks `policy` for
    `samples` action-chunk draws and reduces them to one chunk by their mean or median (`reduce`), answering in the
    protocol's wire format. Calls into the policy are serialised, since Gr00tPolicy is not thread-safe."""
    lock = (
        threading.Lock()
    )  # the server is threaded; Gr00tPolicy is not safe to call from two requests at once (HTTP 500)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
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
                self._send(200, {"ok": True, "server": "gr00t_shim"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/act":
                return self._send(404, {"error": "not found"})
            try:
                req = decode_request(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            except (ProtocolError, ValueError) as e:
                return self._send(400, {"error": str(e)})
            try:
                with lock:
                    action, _info = policy.get_action(to_observation(req, samples))
                self._send(200, encode_response(to_chunk(action, reduce)))
            except Exception as e:  # the client sees a 500 with the reason, the server keeps running
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="HTTP shim around Isaac-GR00T's Gr00tPolicy (untested against real GR00T)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--modality-config-path", default="configs/gr00t/arrival_panda_config.py")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--samples",
        type=int,
        default=1,
        help="chunks sampled per request and reduced to one (GR00T starts each from fresh noise)",
    )
    p.add_argument("--reduce", choices=("mean", "median"), default="mean", help="how --samples chunks are combined")
    p.add_argument(
        "--denoise-steps",
        type=int,
        default=None,
        help="flow-matching steps per chunk; unset keeps the checkpoint's own value (4 for N1.7)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    if a.samples < 1 or (a.denoise_steps is not None and a.denoise_steps < 1):
        p.error("--samples and --denoise-steps must be at least 1")
    policy = load_policy(a.model_path, a.modality_config_path, a.device, a.embodiment_tag)
    if (
        a.denoise_steps is not None
    ):  # the action head reads this attribute on every call (gr00t's open_loop_eval does the same)
        policy.model.action_head.num_inference_timesteps = a.denoise_steps
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(policy, a.samples, a.reduce))
    print(f"gr00t shim at http://{a.host}:{srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
