"""Procedural Panda 7-DoF arm (Franka kinematic chain) + parallel gripper.

Collision uses the original capsule links and hand box (hidden, group 3); the visible arm is built from
primitive Panda-shaped visual geoms (housings, joint rings, Franka hand) that never collide.
"""

from __future__ import annotations

import math

HOME = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]
RANGES = [(-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9), (-3.07, -0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)]
KP = [3000, 3000, 2500, 2500, 1200, 1200, 600]
FINGER_OPEN = 0.05
FINGER_FORCE = 40  # N, each finger
# hand geometry the expert's flap-door plan depends on, per arm (menagerie.HAND is the real hand's):
# flap_bar = how far outside the bar centre the eef holds it, so the pads straddle the bar (the procedural pads sit
# 6.6 mm below the eef, the real ones at it); flap_up = how high the hand backs off after letting the door fall, so
# the wider real hand clears the falling panel; door_z / door_push = close_door push height over the bar and
# standoff of the eef from the bar centre along the door normal
HAND = {"flap_bar": 0.015, "flap_up": 0.15, "door_z": 0.04, "door_push": 0.03}
FPS = 20  # control rate: one action per step
SUBSTEPS = 25  # 0.002 s physics steps per control step
HANDLE_ROOT = 0.022  # grasp point out from a pan's ring: pad half-width 0.015 plus 7 mm
PAD_BELOW_EEF = 0.023  # the finger pads reach this far below the eef site
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
BASE_JOINTS = ["base_x", "base_y", "base_yaw"]  # MAN-3: planar base on robot_base, room-frame x, y, yaw
BASE_SPEED, BASE_TURN = 0.3, 0.5  # m/s and rad/s the expert never exceeds, so the data looks like a slow cart
COL_ATTRS = {"contype": "2", "conaffinity": "1"}  # hits world + moving parts, no self-collision
COL = " ".join(f'{k}="{v}"' for k, v in COL_ATTRS.items())
# eye-in-hand camera on a housing beside the palm, looking along the approach axis and 20 deg inward,
# so the fingertips and whatever is between them sit in the lower half of the frame
WRIST_CAMERA = {"name": "wrist", "pos": "0.06 0 0.045", "xyaxes": "0 1 0 0.94 0 0.342", "fovy": "75"}
EGO_CAMERA = '<camera name="ego_view" pos="-0.35 0.12 1.0" xyaxes="0 -1 0 0.62 0 0.78" fovy="70"/>'
HID = 'group="3"'
WHITE, DARK, GREY = "0.94 0.94 0.95 1", "0.13 0.13 0.14 1", "0.55 0.57 0.6 1"


def _vis(kind: str, rgba: str, **kw: str) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<geom type="{kind}" {attrs} rgba="{rgba}" contype="0" conaffinity="0" group="2"/>'


def _cap(rgba: str, r: float, a: str, b: str) -> str:
    return _vis("capsule", rgba, fromto=f"{a} {b}", size=r)


def _cyl(rgba: str, r: float, a: str, b: str) -> str:
    return _vis("cylinder", rgba, fromto=f"{a} {b}", size=r)


def _link(i: int, pos: str, quat: str, geom: str, inner: str) -> str:
    lo, hi = RANGES[i - 1]
    return (
        f'<body name="link{i}" pos="{pos}" quat="{quat}" gravcomp="1">'
        f'<inertial pos="0 0 0" mass="1.5" diaginertia="0.01 0.01 0.01"/>'
        f'<joint name="joint{i}" axis="0 0 1" range="{lo} {hi}" damping="8" armature="0.1"/>'
        f"{geom}{inner}</body>"
    )


def arm_xml(base_pos: tuple[float, float, float], base_yaw_deg: float, mobile: bool = False) -> str:
    """Body tree for the arm on a pedestal; Franka Panda link offsets (d1=.333, d3=.316, a=.0825, d5=.384, a7=.088).
    mobile=True puts the pedestal on two slide joints and a hinge (room-frame x, y, yaw, zero at the parked pose)."""
    cap = f'type="capsule" rgba="0.92 0.92 0.95 1" {HID} ' + COL
    z0 = "0 0 0"
    hand_vis = (
        _cyl(DARK, 0.031, z0, "0 0 0.012")
        + _vis("box", WHITE, size="0.028 0.075 0.021", pos="0 0 0.033")
        + "".join(_cyl(WHITE, 0.021, f"-0.028 {y} 0.033", f"0.028 {y} 0.033") for y in (-0.075, 0.075))
    )
    hand = (
        f'<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834" gravcomp="1">'
        f'<inertial pos="0 0 0.03" mass="0.6" diaginertia="0.002 0.002 0.002"/>'
        f'<geom type="box" size="0.03 0.09 0.025" pos="0 0 0.03" rgba="0.2 0.2 0.22 1" {HID} {COL}/>'
        + hand_vis
        + _vis("box", DARK, size="0.016 0.012 0.01", pos="0.044 0 0.03")
        + "<camera "
        + " ".join(f'{k}="{v}"' for k, v in WRIST_CAMERA.items())
        + "/>"
        + f'<site name="eef" pos="0 0 0.115" size="0.01" rgba="1 0 0 0"/>'
        + "".join(
            f'<body name="finger_{s}" pos="0 0 0.0584" gravcomp="1">'
            f'<inertial pos="0 0 0.03" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>'
            f'<joint name="finger_{s}" type="slide" axis="0 {sg} 0" range="0 {FINGER_OPEN}" damping="5"/>'
            f'<geom type="box" size="0.015 0.006 0.03" pos="0 {sg * 0.012} 0.05" rgba="0.1 0.1 0.1 1" '
            f'friction="2.0 0.05 0.05" condim="6" {COL}/>'
            + _vis("box", GREY, size="0.0155 0.007 0.026", pos=f"0 {sg * 0.021} 0.0236")
            + "</body>"
            for s, sg in (("l", 1), ("r", -1))
        )
        + "</body>"
    )
    l7 = _link(
        7,
        "0.088 0 0",
        "1 1 0 0",
        f'<geom {cap} size="0.04" fromto="0 0 -0.02 0 0 0.09"/>'
        + _cyl(WHITE, 0.043, "0 0 -0.04", "0 0 0.07")
        + _cyl(DARK, 0.038, "0 0 0.07", "0 0 0.107"),
        hand,
    )
    l6 = _link(
        6,
        "0 0 0",
        "1 1 0 0",
        f'<geom {cap} size="0.045" fromto="0 0 -0.03 0.088 0 0"/>'
        + _cyl(WHITE, 0.052, "0 0 -0.07", "0 0 0.03")
        + _cyl(DARK, 0.042, "0 0 -0.078", "0 0 -0.07")
        + _cap(WHITE, 0.045, z0, "0.088 0 0"),
        l7,
    )
    l5 = _link(
        5,
        "-0.0825 0.384 0",
        "1 -1 0 0",
        f'<geom {cap} size="0.05" fromto="0 0 -0.26 0 0 -0.02"/>'
        + _cyl(WHITE, 0.048, "0 0 -0.23", "0 0 -0.07")
        + _cyl(DARK, 0.057, "0 0 -0.235", "0 0 -0.225"),
        l6,
    )
    l4 = _link(
        4,
        "0.0825 0 0",
        "1 1 0 0",
        f'<geom {cap} size="0.05" fromto="0 0 0 -0.0825 0.384 0"/>'
        + _cyl(WHITE, 0.062, "0 0 -0.062", "0 0 0.062")
        + _cyl(DARK, 0.048, "0 0 0.062", "0 0 0.07")
        + _cap(WHITE, 0.055, z0, "-0.0825 0.05 0")
        + _cyl(WHITE, 0.056, "-0.0825 0.05 0", "-0.0825 0.16 0"),
        l5,
    )
    l3 = _link(
        3,
        "0 -0.316 0",
        "1 1 0 0",
        f'<geom {cap} size="0.05" fromto="0 0 -0.1 0.0825 0 0"/>'
        + _cyl(WHITE, 0.058, "0 0 -0.17", "0 0 -0.03")
        + _cyl(DARK, 0.059, "0 0 -0.17", "0 0 -0.16")
        + _cap(WHITE, 0.055, "0 0 -0.03", "0.0825 0 0"),
        l4,
    )
    l2 = _link(
        2,
        "0 0 0",
        "1 -1 0 0",
        f'<geom {cap} size="0.05" fromto="0 0 0 0 -0.316 0"/>'
        + _cyl(WHITE, 0.065, "0 0 -0.065", "0 0 0.065")
        + _cyl(DARK, 0.05, "0 0 0.065", "0 0 0.073")
        + _cyl(WHITE, 0.058, z0, "0 -0.18 0"),
        l3,
    )
    l1 = _link(
        1,
        "0 0 0.333",
        "1 0 0 0",
        f'<geom {cap} size="0.05" fromto="0 0 -0.2 0 0 0"/>' + _cyl(WHITE, 0.06, "0 0 -0.205", "0 0 -0.03"),
        l2,
    )
    x, y, z = base_pos
    base_joints = base_joints_xml(base_yaw_deg) if mobile else ""
    column = column_xml(z, mobile)
    return (
        f'<body name="robot_base" pos="{x} {y} {z}" euler="0 0 {math.radians(base_yaw_deg)}">'
        + base_joints
        + column
        + f'<geom type="cylinder" size="0.08 0.05" pos="0 0 0.05" rgba="0.92 0.92 0.95 1" {HID}/>'
        + _vis("cylinder", GREY, size="0.13 0.004", pos="0 0 0")
        + plate_xml(z)
        + _vis("cylinder", WHITE, size="0.09 0.06", pos="0 0 0.06")
        + _vis("box", WHITE, size="0.06 0.075 0.045", pos="-0.075 0 0.045")
        + _vis("cylinder", DARK, size="0.062 0.006", pos="0 0 0.126")
        + EGO_CAMERA
        + f"{l1}</body>"
    )


def base_joints_xml(base_yaw_deg: float) -> str:
    """Planar base joints for robot_base. Axes are in the body frame, which is yawed: rotate world x and y back into
    it; the hinge comes last so the slides stay room-aligned."""
    c, s = math.cos(math.radians(base_yaw_deg)), math.sin(math.radians(base_yaw_deg))
    return (
        f'<joint name="base_x" type="slide" axis="{c} {-s} 0" damping="300"/>'
        f'<joint name="base_y" type="slide" axis="{s} {c} 0" damping="300"/>'
        f'<joint name="base_yaw" type="hinge" axis="0 0 1" damping="60"/>'
    )


def column_xml(z: float, mobile: bool = False) -> str:
    """Pedestal column to the floor. A mobile one floats 5 mm so it never grinds on the floor plane."""
    if mobile:
        return (
            f'<geom type="cylinder" size="0.12 {z / 2 - 0.0025}" pos="0 0 {-z / 2 + 0.0025}" rgba="0.25 0.27 0.3 1"/>'
        )
    return f'<geom type="cylinder" size="0.12 {z / 2}" pos="0 0 {-z / 2}" rgba="0.25 0.27 0.3 1"/>'


def plate_xml(z: float) -> str:
    return _vis("cylinder", DARK, size="0.2 0.01", pos=f"0 0 {-z + 0.01}")


def pedestal_xml(z: float, base_yaw_deg: float = 0.0, mobile: bool = False) -> str:
    """Joints, column, floor plate and ego camera for a mount that brings its own arm (menagerie.py), as children of
    robot_base. The joints must come first in the body."""
    return (base_joints_xml(base_yaw_deg) if mobile else "") + column_xml(z, mobile) + plate_xml(z) + EGO_CAMERA


def actuators_xml(mobile: bool = False, finger_open: float = FINGER_OPEN, finger_force: float = FINGER_FORCE) -> str:
    """MJCF position actuators: one per arm joint, one per finger and, with a mobile base, the base x, y and yaw servos
    (stiffer in yaw so the arm cannot turn the base)."""
    arm = "".join(
        f'<position name="a{i + 1}" joint="joint{i + 1}" kp="{KP[i]}" kv="{KP[i] / 15}" '
        f'ctrlrange="{RANGES[i][0]} {RANGES[i][1]}"/>'
        for i in range(7)
    )
    fing = "".join(
        f'<position name="g_{s}" joint="finger_{s}" kp="800" kv="30" ctrlrange="0 {finger_open}" '
        f'forcerange="-{finger_force} {finger_force}"/>'
        for s in ("l", "r")
    )
    base = ""
    if mobile:  # ctrl is the joint value, i.e. the displacement from the parked pose
        # The yaw servo has to be much stiffer than joint 1, which turns the arm against it. At kp 3000 it was exactly
        # as stiff, so each wipe stroke turned the base up to 0.053 rad the other way (28 mm at the sponge), and on
        # u_kitchen_42's 0.90 m counter the sponge travelled 0.27 m pressed, under the 0.4 m check (0.62 m fixed).
        # Ten times kp, kv by sqrt(10) for the same damping ratio (1.6 at 1.3 kg m2): 5.4 mrad, 0.49 to 0.58 m
        # wiped. x, y stay as they are (17 mm of slide in that wipe): ten times those as well wipes 0.60 to 0.72 m
        # but changes how the base tracks a counter_to_counter drive, and with people walking that moved the hero
        # kitchen's seed-1 episode 0023 into a stand-off with a walker it did not meet before (1 of 3, was 2 of 3)
        base = (
            f'<position name="b_x" joint="base_x" kp="20000" kv="1500" ctrlrange="-10 10"/>'
            f'<position name="b_y" joint="base_y" kp="20000" kv="1500" ctrlrange="-10 10"/>'
            f'<position name="b_yaw" joint="base_yaw" kp="30000" kv="630" ctrlrange="-6.3 6.3"/>'
        )
    return arm + fing + base
