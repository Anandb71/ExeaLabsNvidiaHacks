"""Offline check of a twin's RoboCasa-subset layout.yaml against its twin.json (no RoboCasa import).

    python scripts/check_layout.py out/twins/hero_kitchen [--json]
    python scripts/check_layout.py --room hero_kitchen        # compile from scans/ in memory
    python scripts/check_layout.py --room rotated_kitchen --scans tests/fixtures/scans
    python scripts/check_layout.py --roomplan tests/fixtures/roomplan/kitchen_real_01.json

This compares the YAML with the twin it was emitted from. It does NOT prove RoboCasa loads the file.
Checks: frame and units, full sizes (not half-extents), position within 5 cm, yaw as z_rot, wall endpoints
and thickness recovered from pos/size/z_rot, walls counterclockwise (interior on the left), floor covers the
walls, and no fixture or wall dropped, duplicated or invented. Doors, windows, openings and objects are not
written to layout.yaml by design; they are listed as declared omissions, not as errors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TOL = 0.05
DECLARED_OMISSIONS = ("doors", "windows", "openings", "objects")


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _close(a, b, tol=TOL) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def check_layout(twin: dict, layout: dict) -> dict:
    errors: list[str] = []
    err = errors.append
    if twin.get("frame") != "z_up":
        err(f"twin frame is {twin.get('frame')!r}, layout assumes z_up")
    if twin.get("units") != "meters":
        err(f"twin units are {twin.get('units')!r}, layout assumes meters")
    room = layout.get("room") or {}
    group = layout.get("main_group") or {}
    if (
        list(group.get("group_pos", [])) != [0.0, 0.0]
        or group.get("group_z_rot") != 0.0
        or list(group.get("group_origin", [])) != [0, 0]
    ):
        err("main_group is not the identity transform, so pos is not the world frame")

    # fixtures
    rows = group.get("bottom_row") or []
    names = [r.get("name") for r in rows]
    for n in sorted({n for n in names if names.count(n) > 1}):
        err(f"fixture name {n!r} appears more than once in the YAML")
    by_name = {r.get("name"): r for r in rows}
    twin_fx = {f["id"]: f for f in twin.get("fixtures", [])}
    for fid in sorted(twin_fx.keys() - by_name.keys()):
        err(f"fixture {fid} is in the twin but dropped from the YAML")
    for n in sorted(by_name.keys() - twin_fx.keys()):
        err(f"YAML row {n} has no twin fixture")
    for fid, f in twin_fx.items():
        r = by_name.get(fid)
        if r is None:
            continue
        if r.get("type") != f["type"]:
            err(f"{fid}: type {r.get('type')} != twin {f['type']}")
        size, pos = r.get("size", []), r.get("pos", [])
        if not _close(size, f["size"]):
            half = _close(size, [v / 2 for v in f["size"]]) or _close([v / 2 for v in size], f["size"])
            err(f"{fid}: size {size} != twin full size {f['size']}" + (" (half-extent confusion)" if half else ""))
        if not _close(pos, f["position"]):
            err(f"{fid}: pos {pos} is more than 5 cm from twin position {f['position']}")
        if _angle_diff(float(r.get("z_rot", 0.0)), float(f.get("yaw", 0.0))) > 1e-3:
            err(f"{fid}: z_rot {r.get('z_rot', 0.0)} != twin yaw {f.get('yaw', 0.0)}")
        if bool(r.get("island")) != bool(f.get("island")):
            err(f"{fid}: island flag differs")
        if (r.get("default_config_name") or None) != (f.get("default_config_name") or None):
            err(f"{fid}: default_config_name differs")
        if not all(math.isfinite(v) and v > 0 for v in size):
            err(f"{fid}: non-positive or non-finite size {size}")

    # walls
    warnings: list[str] = []
    wrows = {w.get("name"): w for w in room.get("walls") or []}
    twalls = {w["id"]: w for w in twin.get("walls", [])}
    for wid in sorted(twalls.keys() - wrows.keys()):
        err(f"wall {wid} is in the twin but dropped from the YAML")
    for n in sorted(wrows.keys() - twalls.keys()):
        err(f"YAML wall {n} has no twin wall")
    pts = []
    signed_area = 0.0
    for w in twalls.values():
        (x0, y0), (x1, y1) = w["start"], w["end"]
        pts += [(x0, y0), (x1, y1)]
        signed_area += x0 * y1 - x1 * y0
    for wid, w in twalls.items():
        r = wrows.get(wid)
        if r is None:
            continue
        (x0, y0), (x1, y1) = w["start"], w["end"]
        sx, sy, sz = r["size"]
        px, py, pz = r["pos"]
        if "z_rot" in r:
            ang, length, thick = r["z_rot"], sx, sy
        elif sx >= sy:
            ang, length, thick = 0.0, sx, sy
        else:
            ang, length, thick = math.pi / 2, sy, sx
        a = (px - length / 2 * math.cos(ang), py - length / 2 * math.sin(ang))
        b = (px + length / 2 * math.cos(ang), py + length / 2 * math.sin(ang))
        want = [x0, y0, x1, y1]
        if not (_close(list(a) + list(b), want) or _close(list(b) + list(a), want)):
            err(f"{wid}: wall ends recovered from the YAML {a}->{b} do not match twin {(x0, y0)}->{(x1, y1)}")
        if abs(thick - w["thickness"]) > 1e-3:
            err(f"{wid}: thickness {thick} != twin {w['thickness']} (length and thickness swapped?)")
        if abs(sz - w["height"]) > 1e-3 or abs(pz - w["height"] / 2) > 1e-3:
            err(f"{wid}: height {sz} / centre z {pz} do not match twin height {w['height']}")
    if signed_area < 0 and twalls:
        warnings.append(
            "walls run clockwise; docs say counterclockwise. The YAML does not depend on winding "
            "(wall boxes are centred on the segment), but anything using 'interior is left' would flip"
        )

    # floor
    floor = (room.get("floor") or [None])[0]
    if pts and floor is None:
        err("twin has walls but the YAML has no floor")
    if floor is not None:
        fx_, fy_ = floor["pos"][:2]
        hx, hy = floor["size"][0] / 2, floor["size"][1] / 2
        if pts and any(
            not (fx_ - hx - TOL <= x <= fx_ + hx + TOL and fy_ - hy - TOL <= y <= fy_ + hy + TOL) for x, y in pts
        ):
            err("a wall end lies outside the YAML floor")
        for fid, f in twin_fx.items():
            x, y = f["position"][:2]
            if not (fx_ - hx <= x <= fx_ + hx and fy_ - hy <= y <= fy_ + hy):
                warnings.append(f"{fid} centre {[x, y]} is outside the floor rectangle")
    omitted = {k: len(twin.get(k, [])) for k in DECLARED_OMISSIONS if twin.get(k)}
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "declared_omissions": omitted,
        "fixtures": len(twin_fx),
        "walls": len(twalls),
        "scope": "checked against the twin only; not loaded in RoboCasa",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("twin_dir", nargs="?", help="folder with twin.json and layout.yaml")
    p.add_argument("--room", help="compile scans/<room> in memory instead of reading a folder")
    p.add_argument("--scans", default="scans", help="folder containing scans/<room>/scan.json")
    p.add_argument("--roomplan", help="compile and check a RoomPlan export JSON directly")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    inputs = [bool(a.twin_dir), bool(a.room), bool(a.roomplan)]
    if sum(inputs) != 1:
        p.error("give exactly one of twin folder, --room, or --roomplan")
    src = None  # the file being read, named when it cannot be used
    try:
        if a.roomplan:
            from arrival.scan import roomplan
            from arrival.twin.compile import compile_twin, layout_dict

            rp = src = Path(a.roomplan)
            twin = compile_twin(roomplan.load(rp, rp.stem))
            layout = yaml.safe_load(yaml.safe_dump(layout_dict(twin), sort_keys=False))
        elif a.room:
            from arrival.scan.bundle import resolve_scans
            from arrival.twin.compile import layout_dict, room_twin

            scans = resolve_scans(a.scans)
            src = scans / a.room / "scan.json"
            if not src.is_file() and (scans / a.room / "roomplan.json").is_file():
                src = scans / a.room / "roomplan.json"  # a phone capture, built as `sim` builds it
            twin = room_twin(a.room, scans)
            layout = yaml.safe_load(yaml.safe_dump(layout_dict(twin), sort_keys=False))
        else:
            d = Path(a.twin_dir)
            src = d / "twin.json"
            twin = json.loads(src.read_text())
            src = d / "layout.yaml"
            layout = yaml.safe_load(src.read_text())
    except FileNotFoundError as e:
        if getattr(a, "room", None):
            from arrival.scan.bundle import resolve_scans, rooms_hint

            scans = resolve_scans(a.scans)
            p.error(f"no scan for room {a.room!r} under {scans} ({rooms_hint(a.room, scans)}): {e}")
        p.error(str(e))
    except (ValueError, yaml.YAMLError) as e:
        # refused by the scan checks (ScanBundleError), or not JSON or YAML: an input that cannot be checked exits 2,
        # like a missing one; it used to exit 1 with a traceback, the code of a layout that failed the check
        p.error(f"{src}: {e}")
    res = check_layout(twin, layout)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"layout check ({res['fixtures']} fixtures, {res['walls']} walls): {'PASS' if res['ok'] else 'FAIL'}")
        for line in res["errors"]:
            print("  ERROR", line)
        for line in res["warnings"]:
            print("  warn ", line)
        if res["declared_omissions"]:
            print("  not written to layout.yaml by design:", res["declared_omissions"])
        print("  scope:", res["scope"])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
