"""Where the moving parts of a twin's fixtures are: drawer stacks, cabinet doors, wall-cabinet doors, the fridge door
and sinks set into counters. Pure geometry on twin fixture dicts, no MJCF: the scene builder, the stand rules
(preflight), the walkers (people), the task list and the policies all read the same numbers from here."""

from __future__ import annotations

import math

import numpy as np

from arrival.twin.compile import is_counter_host, to_local

APPLIANCES = ("fridge", "stove", "dishwasher")  # floor-standing: counters are trimmed where they run into one
FRONT_COVER = 0.05  # m: a fixture standing within this of a cabinet's front face covers that part of it
FRONT_END = 0.02  # m: a cover reaching this close to an end of the front covers it to that end
MIN_DOOR_W = 0.30  # narrowest cabinet door drawn; a cabinet too narrow for one beside the drawers gets no drawers
DRAWER_W = 0.5  # width of a counter's drawer stack
REVEAL = 0.003  # gap around a drawer front; the door beside the stack ends at it
UPPER_DOOR_MAX = 0.60  # widest wall-cabinet door; a longer run is split into more doors


def sinks_in(counter: dict, sinks: list[dict]) -> list[dict]:
    """The sinks whose centre lies inside the counter's footprint."""
    sx, sy, _ = counter["size"]
    inside = []
    for sink in sinks:
        x, y = to_local(sink["position"], counter)
        if abs(x) < sx / 2 and abs(y) < sy / 2:
            inside.append(sink)
    return inside


def host_of(s: dict, fixtures: list[dict]) -> dict | None:
    """The counter or counter-height cabinet a sink is set into, if any."""
    return next((f for f in fixtures if is_counter_host(f) and sinks_in(f, [s])), None)


def _footprint_box(f: dict, c: dict) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1): the bounding box of fixture f's footprint in counter c's local frame."""
    hx, hy = f["size"][0] / 2, f["size"][1] / 2
    yaw = f.get("yaw", 0.0)
    co, si = math.cos(yaw), math.sin(yaw)
    pts = [
        to_local((f["position"][0] + co * dx - si * dy, f["position"][1] + si * dx + co * dy), c)
        for dx in (-hx, hx)
        for dy in (-hy, hy)
    ]
    return min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts)


def counter_span(c: dict, fixtures: list[dict]) -> tuple[float, float]:
    """Local x-extent of a counter, trimmed at an end that runs into a fridge/stove footprint.
    Appliances wholly inside the span are left alone (the twin data should not place them there; TWN-6)."""
    sx, sy = c["size"][0], c["size"][1]
    x0, x1 = -sx / 2, sx / 2
    for f in fixtures:
        if not f["type"].startswith(APPLIANCES):
            continue
        ax0, ax1, ay0, ay1 = _footprint_box(f, c)
        if ay1 <= -sy / 2 or ay0 >= sy / 2:
            continue
        if ax0 <= x0 < ax1:
            x0 = ax1
        elif ax0 < x1 <= ax1:
            x1 = ax0
    return x0, x1


def front_span(c: dict, fixtures: list[dict]) -> tuple[float, float]:
    """Local x-extent of a cabinet's front that no other fixture stands against. Where two runs meet in a corner,
    the cabinet on the other wall stands against the end of this one's front, and that blind part can hold no door
    or drawer: kitchen_real_01's corner cabinet drew its door there, and the neighbour stopped it at 0.8 deg."""
    sx, sy = c["size"][0], c["size"][1]
    x0, x1 = -sx / 2, sx / 2
    for f in fixtures:
        if f["id"] == c["id"]:
            continue
        ax0, ax1, ay0, ay1 = _footprint_box(f, c)
        if ay1 <= sy / 2 + 0.01 or ay0 >= sy / 2 + FRONT_COVER:
            continue  # flush with the front (the same run), behind it, or clear of it
        lo, hi = max(ax0, x0), min(ax1, x1)
        if hi - lo <= 0.01:
            continue
        # a scanned neighbour stops a hair short of the shared end (kitchen_real_01: 1e-4 m), so "at the end" is
        # within FRONT_END of it; a cover in the middle of the front is left alone
        if lo <= x0 + FRONT_END:
            x0 = hi
        elif hi >= x1 - FRONT_END:
            x1 = lo
    return x0, x1


def drawers_at_right(c: dict, fixtures: list[dict]) -> bool:
    """True when a blind corner takes the right end of the front, where the door's hinge would go. A door hinged
    beside the neighbouring run turns its handle side into it: on kitchen_real_01's corner cabinet the hand met the
    neighbour with the door at 59 deg. The drawer stack then takes that end and the door the open left end, still
    hinged on its right edge, so it swings away from the corner."""
    x0, x1 = front_span(c, fixtures)
    sx = c["size"][0]
    return x1 < sx / 2 - 1e-6 and x0 <= -sx / 2 + 1e-6


def drawer_x(c: dict, fixtures: list[dict]) -> float:
    """Local x of the drawer stack: left quarter, or the middle if a fridge/stove in the same run overlaps the
    left quarter. Judged in the counter's frame, so a run on any wall gets the same layout. The stack stays on the
    open front (front_span), against a blind corner at its right end (drawers_at_right)."""
    x0, x1 = front_span(c, fixtures)
    if drawers_at_right(c, fixtures):
        return x1 - DRAWER_W / 2
    x = -c["size"][0] / 4
    for f in fixtures:
        if not f["type"].startswith(APPLIANCES):
            continue
        rel = float(f.get("yaw", 0.0)) - float(c.get("yaw", 0.0))
        co, si = abs(math.cos(rel)), abs(math.sin(rel))
        lx, ly = to_local(f["position"], c)
        if abs(ly) > (c["size"][1] + si * f["size"][0] + co * f["size"][1]) / 2:
            continue  # across the room, not in this run
        if abs(lx - x) < (co * f["size"][0] + si * f["size"][1]) / 2 + DRAWER_W / 2:
            x = 0.0
            break
    return max(x, x0 + DRAWER_W / 2)


def _door_beside_drawers(c: dict, fixtures: list[dict]) -> tuple[float, float]:
    """(left, hinge) local x of the door beside the drawer stack: the right half of the open front, shrunk so it never
    covers the stack, or with drawers_at_right the open front left of the stack."""
    x0, x1 = front_span(c, fixtures)
    dx = drawer_x(c, fixtures)
    if drawers_at_right(c, fixtures):
        return x0 + 0.02, dx - DRAWER_W / 2 - REVEAL + 0.001  # the panel ends at the stack's reveal
    return max((x0 + x1) / 2 + 0.02, dx + DRAWER_W / 2 + 0.02), x1 - 0.02


def has_drawers(c: dict, fixtures: list[dict]) -> bool:
    """False when the open front cannot hold the drawer stack and a door of MIN_DOOR_W beside it (kitchen_real_01's
    0.43 m cabinet gave the door a width of -0.07 m). Such a cabinet gets one door across its open front."""
    x0, x1 = front_span(c, fixtures)
    dx = drawer_x(c, fixtures)
    left, hinge = _door_beside_drawers(c, fixtures)
    return x0 - 1e-6 <= dx - DRAWER_W / 2 and dx + DRAWER_W / 2 <= x1 + 1e-6 and hinge - left >= MIN_DOOR_W - 1e-6


def has_door(c: dict, fixtures: list[dict]) -> bool:
    return door_w(c, fixtures) >= MIN_DOOR_W - 1e-6


def front_stand(c: dict, family: str, fixtures: list[dict]) -> tuple[float, float]:
    """Where the base stands for a drawer or door task: local x along the run, and distance out from the front."""
    if family in ("open_drawer", "close_drawer"):
        return drawer_x(c, fixtures), 0.72
    # just left of the handle (away from the hinge), outside the door's swing, far enough out that the far end of
    # the handle arc stays within reach of the tilted-down grasp
    return door_hinge_x(c, fixtures) - door_w(c, fixtures) + 0.06 - 0.1, 0.65


def door_hinge_x(c: dict, fixtures: list[dict]) -> float:
    """Local x of the cabinet door's hinge, on the door's right edge: 2 cm in from the right end of the open front,
    or beside the drawer stack when it holds that end (drawers_at_right)."""
    if has_drawers(c, fixtures):
        return _door_beside_drawers(c, fixtures)[1]
    return front_span(c, fixtures)[1] - 0.02


def door_w(c: dict, fixtures: list[dict]) -> float:
    """Cabinet door width: right half of the open front, shrunk so it never covers the drawer stack (or left of the
    stack, drawers_at_right), or the whole open front on a cabinet without drawers."""
    if has_drawers(c, fixtures):
        left, hinge = _door_beside_drawers(c, fixtures)
        return hinge - left
    x0, x1 = front_span(c, fixtures)
    return (x1 - 0.02) - (x0 + 0.02)


def upper_doors(c: dict, fixtures: list[dict] = ()) -> list[tuple[float, float, bool]]:
    """(x_left, x_right, hinge_on_left) per wall-cabinet door over a counter run. Tables, islands and low
    counters get none."""
    if c.get("default_config_name") == "dining" or c.get("island") or c["size"][2] < 0.8:
        return []
    x0, x1 = counter_span(c, list(fixtures))
    x0, x1 = x0 + 0.03, x1 - 0.03  # a reveal at each end keeps the run inside the walls (wall boxes are 5 cm thick)
    n = max(1, math.ceil((x1 - x0) / UPPER_DOOR_MAX))
    w = (x1 - x0) / n
    return [(x0 + k * w, x0 + (k + 1) * w, k % 2 == 0) for k in range(n)]


def is_table(f: dict) -> bool:
    """A scanned table: compiled as a counter for RoboCasa (default_config_name "dining") but built as a table."""
    return f.get("type") == "counter" and f.get("default_config_name") == "dining"


def fridge_mirror(f: dict) -> int:
    """+1 for the default left-hinged fridge, -1 when the fixture says `hinge: "right"` (the door is the mirror
    image in the fixture's x: hinge on the right edge, free edge on the left, swinging the other way)."""
    return -1 if f.get("hinge") == "right" else 1


def fridge_door_frame(f: dict) -> dict:
    """The fridge door in the fixture's frame: hinge xy (left edge, just behind the front), the panel's length
    along local +x when closed, the handle in the door's frame, and the front face's local y. people.py drives
    the door from these numbers, so they live in one place."""
    sx, sy, h = f["size"]
    m = fridge_mirror(f)
    return {
        "hinge": (-m * sx / 2, sy / 2 - 0.03),
        "length": sx,
        "handle": (m * (sx - 0.07), 0.09, h * 0.57),
        "front": sy / 2,
        "mirror": m,
    }


def point_segment_distance(p, a, b) -> float:
    """Distance from point p to the segment a-b (vectors of the same length)."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))
