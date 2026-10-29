"""Render the ~30 s Arrival intro video from the real code.

Everything on screen comes from running the repo: the floor plan is drawn from the compiled twin,
the 3D footage is MuJoCo rendering of real scripted-expert episodes (physics-checked), the joint
readout is the recorded robot state, and the terminal card is captured stdout of the real CLI.

    .venv/bin/python scripts/make_demo_video.py

Writes docs/media/demo.mp4, docs/media/demo.gif and docs/media/poster.png.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import SimEnv  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.tasks import make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

W, H = 960, 540
FPS = 30
MEDIA = ROOT / "docs" / "media"
ROOM = "hero_kitchen"
PY = sys.executable
VIDEO_OUT = "out/demo_video"

BG = (14, 17, 22)
FG = (240, 242, 245)
DIM = (150, 158, 170)
ACCENT = (118, 185, 0)  # NVIDIA-ish green


# ---------------------------------------------------------------- fonts / captions
def _font(size: int, mono: bool = False, bold: bool = False):
    """SF (has the → glyph) first, then Helvetica / DejaVu, then PIL's default."""
    cands = (
        ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf"]
        if mono
        else ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"]
    )
    for c in cands:
        try:
            f = ImageFont.truetype(c, size)
        except OSError:
            continue
        try:
            f.set_variation_by_name("Bold" if bold else ("Regular" if not mono else "Regular"))
        except Exception:  # not a variable font
            pass
        return f
    return ImageFont.load_default()


def caption(img: Image.Image, title: str, sub: str) -> Image.Image:
    img = img.convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    top = H - 92
    d.rectangle([0, top, W, H], fill=(8, 10, 14, 205))
    d.rectangle([0, top, 6, H], fill=ACCENT + (255,))
    d.text((26, top + 14), title, font=_font(28, bold=True), fill=FG)
    d.text((26, top + 54), sub, font=_font(19), fill=(205, 210, 218))
    return Image.alpha_composite(img, ov).convert("RGB")


def centered(d: ImageDraw.ImageDraw, y: int, text: str, font, fill) -> None:
    w = d.textlength(text, font=font)
    d.text(((W - w) / 2, y), text, font=font, fill=fill)


# ---------------------------------------------------------------- 1. title / outro
def title_card() -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, 175, "Arrival", _font(72, bold=True), FG)
    d.rectangle([W / 2 - 60, 272, W / 2 + 60, 276], fill=ACCENT)
    centered(d, 300, "One room scan → robot training data in that room", _font(30), (215, 220, 228))
    centered(d, 460, "Exea Labs · NVIDIA × Nebius hackathon", _font(20), DIM)
    return im


def outro_card() -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, 170, "Join in →", _font(40, bold=True), ACCENT)
    centered(d, 225, "github.com/AvnehSBhatia/ExeaLabsNvidiaHacks", _font(36, bold=True), FG)
    centered(d, 320, "Help wanted: real iPhone scans · mobile base · GR00T fine-tune", _font(23), (205, 210, 218))
    centered(d, 460, "Arrival · Exea Labs", _font(18), DIM)
    return im


# ---------------------------------------------------------------- 2. floor plan from twin.json
FIX_COLORS = {"counter": (150, 112, 70), "sink": (120, 160, 200), "stove": (60, 60, 66), "fridge": (200, 205, 212)}


def floor_plan(twin: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), (24, 28, 35))
    d = ImageDraw.Draw(im)
    xs = [p for w in twin["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in twin["walls"] for p in (w["start"][1], w["end"][1])]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    s = min((W - 160) / (x1 - x0), (H - 92 - 90) / (y1 - y0))
    ox = (W - s * (x1 - x0)) / 2
    oy = 40

    def P(x, y):
        return ox + (x - x0) * s, oy + (y - y0) * s

    # grid
    for gx in np.arange(x0, x1 + 1e-6, 0.5):
        d.line([P(gx, y0), P(gx, y1)], fill=(36, 41, 50))
    for gy in np.arange(y0, y1 + 1e-6, 0.5):
        d.line([P(x0, gy), P(x1, gy)], fill=(36, 41, 50))
    lab = _font(16, bold=True)
    small = _font(14)
    order = {"counter": 0, "stove": 1, "fridge": 2, "sink": 3}
    fx = sorted(twin["fixtures"], key=lambda f: next((v for k, v in order.items() if f["type"].startswith(k)), 0))
    labels = []
    for f in fx:
        (cx, cy, _), (sx, sy, _) = f["position"], f["size"]
        kind = next((k for k in FIX_COLORS if f["type"].startswith(k)), "counter")
        a, b = P(cx - sx / 2, cy - sy / 2), P(cx + sx / 2, cy + sy / 2)
        d.rectangle([a, b], fill=FIX_COLORS[kind], outline=(230, 230, 230), width=2)
        name = f["id"]
        tw = d.textlength(name, font=lab)
        tcol = (20, 20, 20) if kind in ("fridge", "sink") else (245, 245, 245)
        if kind == "counter":
            labels.append((((a[0] + b[0] - tw) / 2, b[1] + 6), name, (225, 200, 160)))
        else:
            labels.append((((a[0] + b[0] - tw) / 2, (a[1] + b[1]) / 2 - 9), name, tcol))
    for xy, name, col in labels:
        d.text(xy, name, font=lab, fill=col)
    for o in twin["objects"]:
        x, y = P(o["position"][0], o["position"][1])
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=ACCENT, outline=(10, 10, 10), width=2)
        d.text((x + 9, y + 2), o["label"] if "label" in o else o["id"], font=small, fill=(225, 255, 190))
    for w in twin["walls"]:
        d.line([P(*w["start"]), P(*w["end"])], fill=(245, 245, 245), width=6)
    d.text(
        (ox, oy + (y1 - y0) * s + 10),
        f"{twin['room_id']} · {x1 - x0:.1f} m × {y1 - y0:.1f} m · {len(twin['fixtures'])} fixtures · "
        f"{len(twin['objects'])} objects   (out/twins/{twin['room_id']}/twin.json)",
        font=small,
        fill=DIM,
    )
    return im


# ---------------------------------------------------------------- 3/4. real sim episodes
def episode_frames(twin: dict, task: dict, cam_cfg: dict, side: int = 300) -> dict:
    """Mirror of arrival.sim.collect.run_episode, also rendering a framed hi-res camera each step."""
    env = SimEnv(twin, task, res=128)
    m = env.model
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    big = mujoco.Renderer(m, H, W)
    small = mujoco.Renderer(m, side, side)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_cfg["lookat"]
    cam.azimuth, cam.elevation = cam_cfg["az"], cam_cfg["el"]
    policy = ScriptedExpert()
    policy.reset(env, task)
    out = {"view": [], "ego_view": [], "wrist": [], "state": []}
    tail, n_est = 0, 180
    try:
        for t in range(400):
            obs = env.obs()
            a = policy.act(obs)
            k = min(1.0, t / (0.45 * n_est))
            k = k * k * (3 - 2 * k)
            cam.distance = cam_cfg["d0"] + (cam_cfg["d1"] - cam_cfg["d0"]) * k
            big.update_scene(env.data, cam)
            out["view"].append(big.render().copy())
            for c in ("ego_view", "wrist"):
                small.update_scene(env.data, c)
                out[c].append(small.render().copy())
            out["state"].append(obs["state"])
            env.step(a)
            if getattr(policy, "done", lambda: False)():
                tail += 1
                if tail > 10:
                    break
        ok, detail = env.success(task)
        out["success"], out["detail"] = ok, detail
    finally:
        big.close()
        small.close()
        env.close()
    print(f"[video] {task['task_id']}: {len(out['view'])} steps success={ok} {detail}")
    return out


def resample(seq: list, n: int) -> list:
    return [seq[int(i)] for i in np.linspace(0, len(seq) - 1, n)]


def sim_frames(ep: dict, n: int, title: str, sub: str, tag: str) -> list[Image.Image]:
    res = []
    for i, f in enumerate(resample(ep["view"], n)):
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        badge = f"{tag}  ·  physics check: {'SUCCESS' if ep['success'] else 'MISS'}" if i > n * 0.8 else tag
        tw = d.textlength(badge, font=_font(17, mono=True))
        d.rounded_rectangle([14, 14, 36 + tw, 46], 6, fill=(8, 10, 14))
        d.text((25, 19), badge, font=_font(17, mono=True), fill=ACCENT if i > n * 0.8 else FG)
        res.append(caption(im, title, sub))
    return res


JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]


def data_frames(ep: dict, n: int) -> list[Image.Image]:
    side = ep["ego_view"][0].shape[0]
    idx = np.linspace(0, len(ep["state"]) - 1, n).astype(int)
    states = np.array(ep["state"])
    lo, hi = states.min(0), states.max(0)
    mono, monob = _font(17, mono=True), _font(15, mono=True)
    res = []
    for i in idx:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = 40
        for k, (c, x) in enumerate((("ego_view", 24), ("wrist", 24 + side + 16))):
            im.paste(Image.fromarray(ep[c][i]), (x, y))
            d.rectangle([x, y, x + side, y + 26], fill=(8, 10, 14))
            d.text((x + 8, y + 4), f"observation.images.{c}", font=monob, fill=FG)
        px = 24 + 2 * side + 40
        d.text((px, y), "observation.state", font=_font(18, mono=True, bold=True), fill=ACCENT)
        d.text((px, y + 24), f"t = {i / 20:5.2f} s   frame {i:3d}", font=monob, fill=DIM)
        s = states[i]
        bw = W - px - 24 - 50 - 100
        for j in range(8):
            yy = y + 62 + j * 30
            name = JOINTS[j] if j < 7 else "grip"
            val = f"{s[j]:+.3f}" if j < 7 else f"{s[j] * 1000:.1f}mm"
            d.text((px, yy), name, font=mono, fill=FG)
            bx = px + 50
            d.rectangle([bx, yy + 4, bx + bw, yy + 18], fill=(36, 41, 50))
            frac = 0.5 if hi[j] - lo[j] < 1e-6 else (s[j] - lo[j]) / (hi[j] - lo[j])
            d.rectangle([bx, yy + 4, bx + max(2, frac * bw), yy + 18], fill=ACCENT if j < 7 else (120, 160, 200))
            d.text((bx + bw + 8, yy), val, font=mono, fill=FG)
        d.text((px, y + 62 + 8 * 30 + 6), "rad (arm) · mm (gripper) · 20 Hz", font=monob, fill=DIM)
        res.append(
            caption(im, "3 · Capture robot data", "Real joint states · physics-checked success · failures dropped")
        )
    return res


# ---------------------------------------------------------------- 5. terminal card from real CLI output
def run_cli() -> list[str]:
    cmds = [
        ["-m", "arrival", "sim", "--episodes", "6", "--people", "2", "--out", VIDEO_OUT],
        ["-m", "arrival", "validate", f"{VIDEO_OUT}/datasets/{ROOM}_sim"],
    ]
    lines = []
    for c in cmds:
        p = subprocess.run([PY, *c], cwd=ROOT, capture_output=True, text=True)
        lines.append("$ python " + " ".join(c))
        lines += [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()]
        print(f"[video] {' '.join(c)} -> exit {p.returncode}")
        if p.returncode:
            raise RuntimeError(f"Demo command failed: {' '.join(c)}")
    return lines


def terminal_frames(lines: list[str], n: int) -> list[Image.Image]:
    f = _font(14, mono=True)
    maxc = int((W - 80) / f.getlength("M"))
    wrapped = []
    for ln in lines:
        while len(ln) > maxc:
            wrapped.append(ln[:maxc])
            ln = "    " + ln[maxc:]
        wrapped.append(ln)
    wrapped = wrapped[-21:]
    res = []
    for k in range(n):
        shown = int(min(1.0, k / (0.55 * n)) * len(wrapped)) + 1
        im = Image.new("RGB", (W, H), (24, 28, 35))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([20, 16, W - 20, H - 104], 10, fill=(10, 12, 16), outline=(60, 66, 78))
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([36 + i * 20, 28, 48 + i * 20, 40], fill=c)
        for i, ln in enumerate(wrapped[:shown]):
            col = ACCENT if ln.startswith("$") else (FG if " OK " in ln or "valid" in ln.lower() else (190, 196, 206))
            if "MISS" in ln:
                col = (230, 150, 90)
            d.text((36, 52 + i * 17.5), ln, font=f, fill=col)
        res.append(
            caption(im, "4 · Validated dataset", "LeRobot v2 · parquet + mp4 + modality.json · validated locally")
        )
    return res


# ---------------------------------------------------------------- assembly
def fade_in(frames: list[Image.Image], k: int = 6) -> list[Image.Image]:
    black = Image.new("RGB", (W, H), BG)
    return [Image.blend(black, f, min(1.0, (i + 1) / k)) if i < k else f for i, f in enumerate(frames)]


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    twin = compile_twin(load_room(ROOM, ROOT / "scans"))

    tasks = make_tasks(twin, ["counter_to_sink", "open_drawer"], 2, seed=0, people=2, log=lambda *a: None)
    sink_t, drawer_t = tasks
    drawer_t["handle"], drawer_t["joint"] = f"{drawer_t['fixture']}_drawer0_handle", f"{drawer_t['fixture']}_drawer0"
    fx = {f["id"]: f for f in twin["fixtures"]}
    sx = (sink_t["start"][0] + fx[sink_t["target"]]["position"][0]) / 2
    sink = episode_frames(twin, sink_t, {"lookat": [sx, 0.45, 0.85], "az": -140, "el": -42, "d0": 3.6, "d1": 2.3})
    dfx = fx[drawer_t["fixture"]]
    drawer = episode_frames(
        twin, drawer_t, {"lookat": [dfx["position"][0] - 0.35, 0.75, 0.6], "az": -40, "el": -42, "d0": 3.4, "d1": 2.2}
    )
    if not (sink["success"] and drawer["success"]):
        raise RuntimeError("A filmed episode failed its physics success check")

    term_lines = run_cli()

    sim_sub = "MuJoCo kitchen · drawers & doors on joints · people walking · 7-DoF arm, friction grasp"
    segments = [
        [title_card()] * int(3 * FPS),
        [
            caption(
                floor_plan(twin), "1 · Room → digital twin", "Authored sample room → walls, fixtures, objects in metres"
            )
        ]
        * int(4 * FPS),
        sim_frames(sink, int(6.5 * FPS), "2 · Simulate the room", sim_sub, f'"{sink_t["instruction"]}"'),
        sim_frames(drawer, int(5.5 * FPS), "2 · Simulate the room", sim_sub, f'"{drawer_t["instruction"]}"'),
        data_frames(sink, int(5 * FPS)),
        terminal_frames(term_lines, int(4 * FPS)),
        [outro_card()] * int(3 * FPS),
    ]
    frames = [f for seg in segments for f in fade_in(seg)]

    mp4 = MEDIA / "demo.mp4"
    with imageio.get_writer(
        mp4,
        fps=FPS,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", "20", "-preset", "slow", "-movflags", "+faststart"],
    ) as w:
        for f in frames:
            w.append_data(np.asarray(f))

    # poster: the mug over the sink, with caption
    sink_seg = segments[2]
    sink_seg[int(len(sink_seg) * 0.62)].save(MEDIA / "poster.png")

    gif_fps = 9
    step = FPS / gif_fps
    gif = [frames[int(i * step)].resize((480, 270), Image.LANCZOS) for i in range(int(len(frames) / step))]
    gif = [g.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    gif[0].save(
        MEDIA / "demo.gif", save_all=True, append_images=gif[1:], duration=int(1000 / gif_fps), loop=0, optimize=True
    )
    for p in ("demo.mp4", "demo.gif", "poster.png"):
        print(f"[video] {MEDIA / p}: {(MEDIA / p).stat().st_size / 1e6:.2f} MB")
    print(f"[video] {len(frames)} frames @ {FPS} fps = {len(frames) / FPS:.1f} s")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
