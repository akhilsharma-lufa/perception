"""Robot dexterity demo — open-loop sequence of joint-space and coord-space moves.

No perception involved. Walks the MyCobot 280 through a tour of poses to
show range of motion and verify motion primitives work end-to-end. Useful as
an extended smoke test before wiring vision into the loop.

SAFETY
------
- Clear ~50 cm of space around the arm before running.
- Keep one hand near the power switch (or the e-stop, if you have one).
- Speed defaults to 25 (slow). You can lower with --speed 15.
- Ctrl+C aborts mid-sequence; the script will try to home + release servos
  before exiting.

Run from the repo root:

    python3 -m perception.demos.robot_dexterity                 # full sequence
    python3 -m perception.demos.robot_dexterity --speed 15       # slower
    python3 -m perception.demos.robot_dexterity --skip-coords    # joint-space only
    python3 -m perception.demos.robot_dexterity --repeat 3       # loop 3 times
    python3 -m perception.demos.robot_dexterity --dry-run        # print only
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional

from perception.control import (
    Gripper,
    GripperSettings,
    MyCobotDriver,
    MyCobotDriverSettings,
)


@dataclass
class JointWaypoint:
    label: str
    angles_deg: list[float]
    gripper_value: Optional[int] = None  # 0=open, 100=closed, None=no change
    dwell_s: float = 0.4


@dataclass
class CoordWaypoint:
    label: str
    coords_mm_deg: list[float]  # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    gripper_value: Optional[int] = None
    dwell_s: float = 0.4


# A measured, conservative tour of the workspace. Values were picked so that the
# arm stays inside its reach and doesn't sweep into the table or its own base.
# Tune to taste.
_JOINT_TOUR: list[JointWaypoint] = [
    JointWaypoint("home",                [0,   0,    0,    0,   0,   0], gripper_value=0, dwell_s=0.6),
    JointWaypoint("base swing right",    [40,  0,    0,    0,   0,   0]),
    JointWaypoint("base swing left",     [-40, 0,    0,    0,   0,   0], gripper_value=80),
    JointWaypoint("base centered",       [0,   0,    0,    0,   0,   0], gripper_value=0),
    JointWaypoint("shoulder forward",    [0,   -35,  0,    0,   0,   0]),
    JointWaypoint("elbow tucked",        [0,   -35, -35,   0,   0,   0]),
    JointWaypoint("wrist pitch -45",     [0,   -35, -35,  -45,  0,   0]),
    JointWaypoint("wrist roll +60",      [0,   -35, -35,  -45,  60,  0]),
    JointWaypoint("wrist yaw +90",       [0,   -35, -35,  -45,  60,  90], gripper_value=80),
    JointWaypoint("reach right + twist", [50,  -35, -35,  -45,  60,  90]),
    JointWaypoint("reach left + twist",  [-50, -35, -35,  -45,  60,  90], gripper_value=0),
    JointWaypoint("salute",              [0,   -55, -45,   0,   90,  0], gripper_value=80),
    JointWaypoint("relax",               [0,   -20,  -20,  0,   0,   0], gripper_value=0),
    JointWaypoint("home",                [0,   0,    0,    0,   0,   0], dwell_s=0.6),
]


def _build_coord_tour(start_xyz_rpy: list[float]) -> list[CoordWaypoint]:
    """Build a small cartesian tour relative to a known starting pose.

    The MyCobot 280 in home pose ends up around (50, -65, +410) mm with the
    gripper pointing forward. We do small deltas around that to show
    cartesian-mode moves work, without venturing near the reach edge.
    """
    x, y, z, rx, ry, rz = start_xyz_rpy[:6]
    return [
        CoordWaypoint("coord origin (home)", [x,      y,      z,      rx, ry, rz]),
        CoordWaypoint("coord +X 60 mm",      [x + 60, y,      z,      rx, ry, rz]),
        CoordWaypoint("coord +Y 60 mm",      [x + 60, y + 60, z,      rx, ry, rz]),
        CoordWaypoint("coord -Z 60 mm",      [x + 60, y + 60, z - 60, rx, ry, rz]),
        CoordWaypoint("coord back to origin",[x,      y,      z,      rx, ry, rz]),
    ]


def _fmt_angles(angles) -> str:
    return "[" + ", ".join(f"{float(a):+6.1f}" for a in angles) + "] deg"


def _fmt_coords(coords) -> str:
    if not coords or len(coords) < 6:
        return "<none>"
    x, y, z, rx, ry, rz = coords[:6]
    return (
        f"xyz=({x:+7.1f},{y:+7.1f},{z:+7.1f}) mm "
        f"rpy=({rx:+6.1f},{ry:+6.1f},{rz:+6.1f}) deg"
    )


def _maybe_apply_gripper(gripper: Optional[Gripper], value: Optional[int]) -> None:
    if gripper is None or value is None:
        return
    gripper.set_width(int(value))


def _run_joint_tour(
    driver: MyCobotDriver,
    gripper: Optional[Gripper],
    speed: int,
    dry_run: bool,
) -> None:
    print("\n=== joint-space tour ===")
    for i, wp in enumerate(_JOINT_TOUR):
        prefix = f"[{i+1:02d}/{len(_JOINT_TOUR):02d}] {wp.label:<24}"
        print(f"  {prefix} angles={_fmt_angles(wp.angles_deg)} "
              f"gripper={'-' if wp.gripper_value is None else wp.gripper_value}")
        if dry_run:
            continue
        driver.send_angles_deg(wp.angles_deg, speed=speed)
        try:
            driver.wait_until_done(strict=False)
        except Exception as e:
            print(f"    wait_until_done: {e} — continuing")
        _maybe_apply_gripper(gripper, wp.gripper_value)
        time.sleep(float(wp.dwell_s))


def _run_coord_tour(
    driver: MyCobotDriver,
    gripper: Optional[Gripper],
    speed: int,
    dry_run: bool,
) -> None:
    print("\n=== cartesian tour ===")
    try:
        start = driver.get_coords_mm_deg()
    except Exception as e:
        print(f"  could not read starting coords: {e}; skipping cartesian tour")
        return
    print(f"  start pose: {_fmt_coords(start)}")
    tour = _build_coord_tour(start)
    for i, wp in enumerate(tour):
        prefix = f"[{i+1:02d}/{len(tour):02d}] {wp.label:<28}"
        print(f"  {prefix} {_fmt_coords(wp.coords_mm_deg)} "
              f"gripper={'-' if wp.gripper_value is None else wp.gripper_value}")
        if dry_run:
            continue
        driver.send_coords_mm_deg(wp.coords_mm_deg, speed=speed)
        try:
            driver.wait_until_done(strict=False, timeout_s=60.0)
        except Exception as e:
            print(f"    wait_until_done: {e} — continuing")
        _maybe_apply_gripper(gripper, wp.gripper_value)
        time.sleep(float(wp.dwell_s))


def main() -> None:
    parser = argparse.ArgumentParser(description="MyCobot 280 dexterity demo")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--speed", type=int, default=25, help="1..100 (default 25, slow)")
    parser.add_argument("--repeat", type=int, default=1, help="run the full sequence N times")
    parser.add_argument("--skip-joints", action="store_true")
    parser.add_argument("--skip-coords", action="store_true")
    parser.add_argument("--skip-gripper", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print only, don't move")
    args = parser.parse_args()

    speed = max(1, min(100, int(args.speed)))
    print(f"[dexterity] connecting to MyCobot 280 on {args.port} @ {args.baudrate}")
    driver = MyCobotDriver(
        MyCobotDriverSettings(port=args.port, baudrate=args.baudrate, default_speed=speed)
    )
    if not args.dry_run:
        driver.connect()
    gripper: Optional[Gripper] = None

    try:
        if not args.dry_run:
            if not driver.is_power_on():
                print("[dexterity] power_on")
                driver.power_on()
            if not args.skip_gripper:
                gripper = Gripper(driver, GripperSettings())
                gripper.open()
        else:
            print("[dexterity] DRY RUN — will print sequence without moving")

        for cycle in range(max(1, int(args.repeat))):
            if int(args.repeat) > 1:
                print(f"\n========== cycle {cycle+1}/{args.repeat} ==========")
            if not args.skip_joints:
                _run_joint_tour(driver, gripper, speed, args.dry_run)
            if not args.skip_coords and not args.dry_run:
                _run_coord_tour(driver, gripper, speed, args.dry_run)
            elif not args.skip_coords and args.dry_run:
                # In dry-run we can't read start pose; show a generic placeholder.
                print("\n=== cartesian tour (dry-run, skipped — needs live start pose) ===")

        print("\n[dexterity] DONE")
    except KeyboardInterrupt:
        print("\n[dexterity] interrupted — attempting safe home + release")
        if not args.dry_run:
            try:
                driver.send_angles_deg([0, 0, 0, 0, 0, 0], speed=speed)
                driver.wait_until_done(timeout_s=8.0)
            except Exception as e:
                print(f"  home attempt failed: {e}")
        sys.exit(130)
    finally:
        if not args.dry_run and driver.is_connected:
            try:
                driver.release_all_servos()
            except Exception:
                pass
            driver.disconnect()


if __name__ == "__main__":
    main()
