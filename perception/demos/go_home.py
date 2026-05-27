"""Reliably park the arm at home and open the gripper — from any pose.

Recovery tool for when the arm gets stuck in an awkward position, or to reset
between runs. Sends the arm to [0,0,0,0,0,0] (arm up, gripper perpendicular) and
opens the gripper, verifying arrival by joint angle (the firmware's is_moving()
is unreliable). Leaves servos engaged so the next run starts from a known pose.

Usage:
    # Arm + gripper home (default).
    python -m perception.demos.go_home --port /dev/ttyUSB0

    # Gripper not attached yet (skip the gripper open).
    python -m perception.demos.go_home --port /dev/ttyUSB0 --no-gripper

    # Be patient with a far-away start / slow speed.
    python -m perception.demos.go_home --speed 20 --timeout-s 25

Exit codes:
    0  arm reached home within tolerance
    1  timed out before reaching home (joints still off)
"""
from __future__ import annotations

import argparse
import sys

from perception.control import (
    Gripper,
    GripperSettings,
    MyCobotDriver,
    MyCobotDriverSettings,
    safe_home,
)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m perception.demos.go_home",
        description="Park the arm at home [0,0,0,0,0,0] and open the gripper, "
                    "from any pose. Verifies arrival by joint angle.",
    )
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=1_000_000)
    p.add_argument("--speed", type=int, default=30,
                   help="Joint speed 1-100 (default 30; lower is gentler).")
    p.add_argument("--timeout-s", type=float, default=15.0,
                   help="Seconds to wait for the arm to reach home.")
    p.add_argument("--tol-deg", type=float, default=3.0,
                   help="Per-joint tolerance (deg) for 'arrived'.")
    p.add_argument("--no-gripper", action="store_true",
                   help="Do not open/talk to the gripper (e.g. not attached yet).")
    args = p.parse_args()

    driver = MyCobotDriver(MyCobotDriverSettings(port=args.port, baudrate=args.baudrate))
    driver.connect()
    try:
        gripper = None if args.no_gripper else Gripper(driver, GripperSettings())
        ok = safe_home(
            driver,
            gripper,
            speed=int(args.speed),
            timeout_s=float(args.timeout_s),
            tol_deg=float(args.tol_deg),
            open_gripper=not args.no_gripper,
        )
    finally:
        # Leave servos engaged on purpose (no release_all_servos): the next run
        # then starts from a known pose. disconnect() does not release.
        driver.disconnect()

    if ok:
        print("[go_home] HOME OK")
        sys.exit(0)
    print("[go_home] HOME FAILED — arm did not reach home within tolerance.",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
