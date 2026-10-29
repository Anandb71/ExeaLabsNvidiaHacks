"""C4 (TWN-5): small objects found in posed RGB-D scan frames, placed in the twin's room frame.

    run = detect_objects(frames_dir, twin, bundle)       # {"state", "objects", "rejected", ...}
    fill_objects(twin, run, seed)                        # arrival.twin.compile: detections, else default objects

`DetectionClient` asks Nemotron Omni on Token Factory (role "omni", the same client and settings as the planner)
which catalog objects one frame shows, each with a pixel box and a confidence. Replies are cached by a hash of the
image and the prompt version, so tests and repeat runs read the cache and make no call. A frame with no cached reply
and no credentials is left unread: nothing is ever guessed. Only frames in the `train` split are read, so the
held-out frames the realism stage scores never shape the twin.

Each accepted box is back-projected with the frame's depth, intrinsics and pose (`arrival.scan.rgbd.unproject`).
Over each counter or table the points at object height, less those within WALL_STRIP_M of a wall's face in the twin and
those on the backsplash strip of a counter or storage worktop, are split into connected groups, and a group counts
only if it could be the catalog object standing on that top (`locate`); the one whose centre projects nearest the box
centre gives x, y, and z sits the catalog-size object on the top. Its footprint must then lie over the top, clear of
walls and of other fixtures, a sink's faucet strip included (`clearance`). Within a frame, two boxes on one spot are one
object; across frames a hit joins the nearest object of its label within DEDUP_M, never one holding its own frame, and
the merged position must pass `clearance` again, since the mean of two clear spots can fall in a keep-out (the free
space around a sink is not convex). A box that is not placed is listed in `rejected` with its reason
(`unsupported_label`, `low_confidence`, `bad_box`, `no_depth`, `no_support`, `nothing_above_top`, `too_wide`,
`not_standing`, `flat_surface`, `off_top`, `against_wall`, `on_fixture`, `same_object`); nothing is dropped silently.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from arrival.catalog.categories import OBJECT_SPECS, RIM_HELD, catalog_id
from arrival.scan import rgbd

PROMPT_VERSION = 1
MIN_CONFIDENCE = 0.5
DEDUP_M = 0.10  # two boxes of one label closer than this in x, y are the same object seen twice
MAX_FRAMES = 12  # frames read per run, evenly spaced over the train split
ABOVE_TOP_M = 0.01  # a point this far above the support top belongs to the object, not the top
SLACK_M = 0.02  # room around the catalog size for size variation and depth noise
CELL_M = 0.02  # points in neighbouring cells of this grid belong to one surface; about two depth pixels at 1 m
WALL_STRIP_M = 0.03  # points this close to a wall face are the wall, not something standing in front of it
BACK_STRIP_M = 0.025  # the back of a counter top, where its backsplash stands
FAUCET_M = 0.10  # behind a sink (its local -y) the faucet stands; the scene puts it 4 cm in from the counter's back
CLEAR_TOL_M = 0.01  # an object may overhang a top edge or touch a wall by this much
MIN_POINTS = 3
DEFAULT_CACHE = None  # the upload worker keeps no reply cache; tests point this at a fixture
ALIASES = {
    "coffee mug": "mug",
    "glass": "cup",
    "dish": "plate",
    "frying pan": "pan",
    "skillet": "pan",
    "water bottle": "bottle",
    "remote control": "remote",
    "tv remote": "remote",
}
SYSTEM_PROMPT = (
    "You find small objects in one RGB photo of a room. Report only these labels: "
    + ", ".join(OBJECT_SPECS)
    + '. Anything else you see on a counter or table is "other:<name>". Return only JSON: {"objects": '
    '[{"label": str, "box": [x0, y0, x1, y1] in pixels of this image, "confidence": 0 to 1}]}, '
    'and {"objects": []} when there is none.'
)


class DetectionClient:
    """Detections for one frame: the cache first, then Token Factory. `read` returns (objects or None, source)."""

    def __init__(self, cache_path: str | Path | None = None, log=print):
        self.log = log
        self.cache_path = Path(cache_path) if cache_path else None
        self.entries: dict = {}
        self.model = None
        self._offline = None  # why the model cannot be called, once known
        if self.cache_path and self.cache_path.is_file():
            self.entries = json.loads(self.cache_path.read_text(encoding="utf-8")).get("entries", {})

    @staticmethod
    def key(rgb_path: Path) -> str:
        h = hashlib.sha256(f"detect-v{PROMPT_VERSION}:".encode())
        h.update(Path(rgb_path).read_bytes())
        return h.hexdigest()[:24]

    def read(self, frame: rgbd.Frame) -> tuple[list | None, str]:
        """The detections in one frame and where they came from: the cached reply when there is one, else a live call to
        the omni model (cached on success). Unconfigured, offline or failed calls return None and the reason."""
        key = self.key(frame.rgb)
        hit = self.entries.get(key)
        if isinstance(hit, dict):
            return parse_reply(hit.get("reply")), hit.get("source", "cache")
        if self._offline:
            return None, self._offline
        from arrival.models.token_factory import TokenFactoryNotConfigured, complete
        from arrival.models.token_factory import config as tf_config

        try:
            from PIL import Image

            with Image.open(frame.rgb) as im:
                w, h = im.size
            data = base64.b64encode(Path(frame.rgb).read_bytes()).decode()
            mime = "image/png" if frame.rgb.suffix.lower() == ".png" else "image/jpeg"
            txt = complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"The image is {w}x{h} pixels."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                        ],
                    },
                ],
                role="omni",
                temperature=0,
            )
            objects = parse_reply(txt)
        except TokenFactoryNotConfigured as e:
            self._offline = "not_configured"
            self.log(f"[detect] Token Factory not configured ({e}); frames without a cached reply are not read")
            return None, self._offline
        except Exception as e:  # noqa: BLE001 - network or reply errors leave the frame unread, loudly
            if isinstance(e, OSError):
                self._offline = "unreachable"
            self.log(f"[detect] frame {frame.id}: model call failed ({type(e).__name__}); frame not read")
            return None, "call_failed"
        self.model = tf_config()["models"].get("omni")
        self.entries[key] = {
            "frame": frame.id,
            "source": "token_factory",
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "reply": {"objects": objects},
        }
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({"entries": self.entries}, indent=2) + "\n", encoding="utf-8")
        return objects, "token_factory"


def parse_reply(reply) -> list:
    """The `objects` list of a reply (JSON text or an already parsed dict). Entries keep only label, box and
    confidence; a reply that is not that shape raises ValueError."""
    if isinstance(reply, str):
        reply, _ = json.JSONDecoder().raw_decode(reply[reply.index("{") :])
    if not isinstance(reply, dict) or not isinstance(reply.get("objects"), list):
        raise ValueError("reply has no objects list")
    return [
        {"label": o.get("label"), "box": o.get("box"), "confidence": o.get("confidence")}
        for o in reply["objects"]
        if isinstance(o, dict)
    ]


def catalog_label(label) -> str | None:
    """The catalog label a model label names, or None for anything the catalog has no spec for."""
    if not isinstance(label, str):
        return None
    name = ALIASES.get(label.strip().lower(), label.strip().lower())
    return name if name in OBJECT_SPECS else None


def _top(f: dict) -> float:
    return f["position"][2] + f["size"][2] / 2


def supports(twin: dict) -> list[dict]:
    """Fixtures a catalog object can stand on: counters (scanned, synthesized, storage at counter height), tables."""
    from arrival.twin.compile import is_counter_host

    return [f for f in twin.get("fixtures", []) if is_counter_host(f) or f.get("source_category") == "table"]


def support_under(xy, twin: dict, below_z: float = math.inf) -> dict | None:
    """The highest counter or table whose footprint holds xy and whose top is under below_z."""
    tops = [f for f in supports(twin) if _over(np.asarray([xy], float), f)[0] and _top(f) < below_z]
    return max(tops, key=_top) if tops else None


def _over(pts: np.ndarray, f: dict) -> np.ndarray:
    """Which points lie over fixture f's footprint in x, y."""
    yaw = float(f.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = pts[:, 0] - f["position"][0], pts[:, 1] - f["position"][1]
    return (np.abs(c * dx + s * dy) <= f["size"][0] / 2) & (np.abs(-s * dx + c * dy) <= f["size"][1] / 2)


def has_backsplash(f: dict) -> bool:
    """Whether top f stands against a wall with a backsplash at its back: a counter or a RoomPlan storage worktop
    (`is_counter_host`), not a table or an island. The scene draws one on an island too, which a real island lacks."""
    from arrival.twin.compile import is_counter_host

    return is_counter_host(f) and not f.get("island") and f.get("default_config_name") != "dining"


def _wall_distance(pts: np.ndarray, w: dict) -> np.ndarray:
    """Horizontal distance from points to a wall's centre line."""
    a, b = np.asarray(w["start"], float), np.asarray(w["end"], float)
    d = b - a
    t = np.clip(((pts[:, :2] - a) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.linalg.norm(pts[:, :2] - (a + t[:, None] * d), axis=1)


def _backdrop(pts: np.ndarray, f: dict, twin: dict) -> np.ndarray:
    """Points that belong to a wall or to the backsplash at the back of top f (local -y; +y is its front)."""
    near = np.zeros(len(pts), bool)
    for w in twin.get("walls", []):
        near |= _wall_distance(pts, w) < float(w.get("thickness", 0.0)) / 2 + WALL_STRIP_M
    if has_backsplash(f):
        yaw = float(f.get("yaw", 0.0))
        dx, dy = pts[:, 0] - f["position"][0], pts[:, 1] - f["position"][1]
        near |= -math.sin(yaw) * dx + math.cos(yaw) * dy < -f["size"][1] / 2 + BACK_STRIP_M
    return near


def clearance(xy, label: str, support: dict, twin: dict) -> str | None:
    """Why an object of `label` centred at xy cannot stand on `support`, or None. Its footprint, a circle of the
    object's smaller half-width since the yaw is not known, must lie over the top (`off_top`), clear of walls
    (`against_wall`), and clear of any fixture it would stand in, a sink with the strip behind it where the faucet
    stands included (`on_fixture`)."""
    r = min(OBJECT_SPECS[label][1][:2]) / 2
    p = np.asarray([[xy[0], xy[1], 0.0]], float)
    yaw = float(support.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xy[0] - support["position"][0], xy[1] - support["position"][1]
    reach = r - CLEAR_TOL_M
    if abs(c * dx + s * dy) > support["size"][0] / 2 - reach or abs(-s * dx + c * dy) > support["size"][1] / 2 - reach:
        return "off_top"
    if any(_wall_distance(p, w)[0] < float(w.get("thickness", 0.0)) / 2 + reach for w in twin.get("walls", [])):
        return "against_wall"
    top, height = _top(support), OBJECT_SPECS[label][1][2]
    for g in twin.get("fixtures", []):
        if g is support or g in supports(twin) or g["position"][2] - g["size"][2] / 2 >= top + height:
            continue  # another top, or hung above the object (a wall cabinet)
        m = r - CLEAR_TOL_M
        back = FAUCET_M if g["type"] == "sink" else 0.0
        gy = float(g.get("yaw", 0.0))
        gc, gs = math.cos(gy), math.sin(gy)
        ex, ey = xy[0] - g["position"][0], xy[1] - g["position"][1]
        lx, ly = gc * ex + gs * ey, -gs * ex + gc * ey
        if abs(lx) < g["size"][0] / 2 + m and -g["size"][1] / 2 - m - back < ly < g["size"][1] / 2 + m:
            return "on_fixture"
    return None


def _thickness(xy: np.ndarray) -> float:
    """Spread of points across their main direction in x, y: about 0 for a wall or backsplash seen face on."""
    c = xy - xy.mean(axis=0)
    return float(np.ptp(c @ np.linalg.svd(c, full_matrices=False)[2][-1]))


def _clusters(xy: np.ndarray, cell: float = CELL_M) -> list[np.ndarray]:
    """Index arrays of the groups of points joined through occupied neighbouring cells of a `cell` grid."""
    cells: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(map(tuple, np.floor(xy / cell).astype(int).tolist())):
        cells.setdefault(key, []).append(i)
    out, seen = [], set()
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        stack, members = [start], []
        while stack:
            cx, cy = stack.pop()
            members += cells[(cx, cy)]
            for nb in ((cx + a, cy + b) for a in (-1, 0, 1) for b in (-1, 0, 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        out.append(np.array(members))
    return out


def locate(
    box,
    label: str,
    depth: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    scale: tuple[float, float],
    twin: dict,
    confidence: np.ndarray | None = None,
) -> tuple[list[float] | None, str | None]:
    """Room-frame centre [x, y, z] of a catalog object from its RGB pixel box, or (None, reason).
    `scale` maps RGB pixels to depth pixels; K is for the depth image."""
    try:
        u0, v0, u1, v1 = (float(b) for b in box)
    except (TypeError, ValueError):
        return None, "bad_box"
    if not all(math.isfinite(x) for x in (u0, v0, u1, v1)):
        return None, "bad_box"
    h, w = depth.shape
    c0, c1 = max(0, math.floor(min(u0, u1) * scale[0])), min(w, math.ceil(max(u0, u1) * scale[0]))
    r0, r1 = max(0, math.floor(min(v0, v1) * scale[1])), min(h, math.ceil(max(v0, v1) * scale[1]))
    if c1 <= c0 or r1 <= r0:
        return None, "bad_box"
    masked = np.zeros_like(depth)
    masked[r0:r1, c0:c1] = depth[r0:r1, c0:c1]
    if confidence is not None:
        masked[confidence < 1] = 0.0  # ARKit confidence 0 is low
    pts = rgbd.unproject(masked, K, T)
    if len(pts) < MIN_POINTS:
        return None, "no_depth"
    size = OBJECT_SPECS[label][1]
    # A box looser than the object also holds what stands behind or beside it: a wall, a backsplash, a stove, another
    # object. For each counter or table, the box's points over its footprint at object height are split into connected
    # groups, and a group counts only if it could be this object standing on that top: no wider than the object, its
    # lowest point near the top, some of the object's height, and depth in x, y (a wall or backsplash seen face on is
    # a line there). Of those, the group whose centre projects nearest the box centre is the object.
    centre_uv = np.array([(u0 + u1) / 2 * scale[0], (v0 + v1) / 2 * scale[1]])
    best, why = None, "no_support"
    for f in supports(twin):
        top, over = _top(f), _over(pts, f)
        if why == "no_support" and (over & (pts[:, 2] > top - 0.005)).any():
            why = "nothing_above_top"
        band = pts[
            over
            & (pts[:, 2] > top + ABOVE_TOP_M)
            & (pts[:, 2] < top + 1.25 * size[2] + SLACK_M)
            & ~_backdrop(pts, f, twin)
        ]
        for g in _clusters(band[:, :2]):
            obj = band[g]
            if len(obj) < MIN_POINTS:
                continue
            if np.ptp(obj[:, :2], axis=0).max() > max(size[:2]) + 2 * SLACK_M:
                fault = "too_wide"
            elif obj[:, 2].min() > top + max(0.03, size[2] / 2) or np.ptp(obj[:, 2]) < 0.4 * size[2]:
                fault = "not_standing"
            elif _thickness(obj[:, :2]) < 0.25 * min(size[:2]):
                fault = "flat_surface"
            else:
                fault = None
            if fault:
                why = fault if why in ("no_support", "nothing_above_top") else why
                continue
            uv, _ = rgbd.project(obj.mean(axis=0, keepdims=True), K, T)
            off = float(np.linalg.norm(uv[0] - centre_uv))
            if best is None or off < best[0]:
                best = (off, obj, f)
    if best is None:
        return None, why
    _, obj, f = best
    # the camera sees the near side only: on a round object of radius r the visible points, weighted by projected
    # area, centre about r * pi / 4 short of the axis, so push the estimate that far away from the camera
    # (a bowl or plate seen from above shows its far inner wall too, so it gets no shift)
    xy = obj[:, :2].mean(axis=0)
    away = xy - np.asarray(T, float)[:2, 3]
    if np.linalg.norm(away) > 1e-6 and label not in RIM_HELD:
        xy = xy + away / np.linalg.norm(away) * (min(size[:2]) / 2) * math.pi / 4
    blocked = clearance(xy, label, f, twin)
    if blocked:
        return None, blocked
    return [float(xy[0]), float(xy[1]), _top(f) + size[2] / 2], None


def _train_frames(m: rgbd.Manifest) -> list[rgbd.Frame]:
    train = [f for f in m.frames if f.split == "train"]
    if len(train) <= MAX_FRAMES:
        return train
    idx = np.linspace(0, len(train) - 1, MAX_FRAMES).round().astype(int)
    return [train[i] for i in sorted(set(idx.tolist()))]


def detect_objects(
    frames_dir, twin: dict, bundle: dict | None = None, client: DetectionClient | None = None, log=print
) -> dict:
    """Read the train frames of a CAP-2a bundle and return the twin objects found, merged across frames.
    Poses map into the twin as the realism stage maps them: through the normalized bundle's `arkit_world_to_room`
    when it has one, else unchanged, and a named scan must name the frames' room. Never raises for a bad bundle:
    the result's `state` is `detected`, `none_found` or `not_run` and `reason` says why when nothing was read."""
    client = client or DetectionClient(DEFAULT_CACHE, log=log)
    result = {
        "state": "not_run",
        "reason": None,
        "frames_read": 0,
        "frames_unread": 0,
        "reply_sources": {},
        "model": None,
        "objects": [],
        "rejected": [],
    }
    try:
        m = rgbd.load_manifest(frames_dir)
    except rgbd.ManifestError as e:
        result["reason"] = f"frames bundle unusable ({e.code})"
        return result
    bundle = bundle or {}
    if bundle.get("source_room_id") and bundle["source_room_id"] != m.room_id:
        result["reason"] = "the scan and frame bundle name different rooms"
        return result
    frames = _train_frames(m)
    if not frames:
        result["reason"] = "no train frames to read"
        return result
    A = np.eye(4)
    if "arkit_world_to_room" in bundle:
        A = np.asarray(bundle["arkit_world_to_room"], float) @ np.linalg.inv(m.arkit_world_to_room)
    scale = (m.depth_size[0] / m.rgb_size[0], m.depth_size[1] / m.rgb_size[1])
    hits: list[dict] = []
    for f in frames:
        try:
            found, source = client.read(f)
        except ValueError:
            found, source = None, "unusable_reply"
        result["reply_sources"][source] = result["reply_sources"].get(source, 0) + 1
        if found is None:
            result["frames_unread"] += 1
            continue
        result["frames_read"] += 1
        depth, K = rgbd.load_depth(m, f), rgbd.depth_intrinsics(m, f)
        conf = None
        if f.confidence is not None:
            from PIL import Image

            conf = np.asarray(Image.open(f.confidence))
        here: list[dict] = []
        for o in found:
            rec = {"frame": f.id, "label": o["label"], "box": o["box"], "confidence": o["confidence"]}
            label = catalog_label(o["label"])
            c = o["confidence"]
            if label is None:
                result["rejected"].append({**rec, "reason": "unsupported_label"})
            elif isinstance(c, bool) or not isinstance(c, (int, float)) or not c >= MIN_CONFIDENCE:
                result["rejected"].append({**rec, "reason": "low_confidence"})
            else:
                pos, why = locate(o["box"], label, depth, K, A @ f.camera_to_room, scale, twin, conf)
                if pos is None:
                    result["rejected"].append({**rec, "reason": why})
                else:
                    here.append({**rec, "label": label, "position": pos})
        # two boxes in one frame that land within the object's footprint of each other found the same object
        kept: list[dict] = []
        for h in sorted(here, key=lambda h: -float(h["confidence"])):
            reach = max(OBJECT_SPECS[h["label"]][1][:2])
            if any(k["label"] == h["label"] and math.dist(k["position"][:2], h["position"][:2]) < reach for k in kept):
                result["rejected"].append(
                    {k: h[k] for k in ("frame", "label", "box", "confidence")} | {"reason": "same_object"}
                )
            else:
                kept.append(h)
        hits += kept
    if result["frames_read"] == 0:
        result["reason"] = "no frame could be read: " + ", ".join(sorted(result["reply_sources"]))
        return result
    result["model"] = client.model
    result["objects"] = merge(hits, twin, result["rejected"])
    result["state"] = "detected" if result["objects"] else "none_found"
    return result


def merge(hits: list[dict], twin: dict, rejected: list | None = None) -> list[dict]:
    """One twin object per cluster of same-label hits (greedy, highest confidence first): a hit joins the nearest
    cluster centre within DEDUP_M that has no hit from its own frame, since one frame never shows one object twice.
    A cluster whose centre has no counter or table under it, or fails `clearance` there, is not placed; its hits go to
    `rejected` with that reason (`no_support`, `off_top`, `against_wall`, `on_fixture`)."""
    clusters: list[list[dict]] = []
    for h in sorted(hits, key=lambda h: (-float(h["confidence"]), h["frame"])):
        near = [
            (math.dist(h["position"][:2], np.mean([x["position"][:2] for x in c], axis=0)), k)
            for k, c in enumerate(clusters)
            if c[0]["label"] == h["label"] and all(x["frame"] != h["frame"] for x in c)
        ]
        near = [(d, k) for d, k in near if d < DEDUP_M]
        if near:
            clusters[min(near)[1]].append(h)
        else:
            clusters.append([h])
    taken = {o["id"] for o in twin.get("objects", [])}
    out = []
    for c in clusters:
        label = c[0]["label"]
        size = list(OBJECT_SPECS[label][1])
        xy = np.mean([x["position"][:2] for x in c], axis=0)
        support = support_under(xy, twin, below_z=float(np.mean([x["position"][2] for x in c])))
        why = "no_support" if support is None else clearance(xy, label, support, twin)
        if why:
            if rejected is not None:
                rejected += [
                    {"frame": x["frame"], "label": label, "box": x["box"], "confidence": x["confidence"], "reason": why}
                    for x in c
                ]
            continue
        z = _top(support) + size[2] / 2
        n = 1
        while f"{label}_{n}" in taken:
            n += 1
        taken.add(f"{label}_{n}")
        out.append(
            {
                "id": f"{label}_{n}",
                "catalog_id": catalog_id(label),
                "label": label,
                "size": size,
                "position": [round(float(xy[0]), 6), round(float(xy[1]), 6), round(z, 6)],
                "yaw": 0.0,
                "placed_by": "detect_objects",
                "provenance": "detected",
                "detection": {
                    "confidence": max(float(x["confidence"]) for x in c),
                    "frames": sorted({x["frame"] for x in c}),
                    "evidence": [{"frame": x["frame"], "box": x["box"], "confidence": x["confidence"]} for x in c],
                },
            }
        )
    return out
