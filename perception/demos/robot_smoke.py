"""Phase-1 sanity check: connect to the MyCobot 280 and exercise the gripper.

Run from the Jetson once pymycobot is installed and the arm is powered on:

    python3 -m perception.demos.robot_smoke --port /dev/ttyAMA0

Optional flags let you skip the home move or the gripper cycle if you're
isolating a particular failure.
"""

from __future__ import annotations

import argparse
import time

from perception.control import (
    Gripper,
    GripperSettings,
    MyCobotDriver,
    MyCobotDriverSettings,
)


def _fmt_coords(coords) -> str:
    if not coords or len(coords) < 6:
        return "<none>"
    x, y, z, rx, ry, rz = coords[:6]
    return (
        f"xyz=({x:+7.1f},{y:+7.1f},{z:+7.1f}) mm  "
        f"rpy=({rx:+6.1f},{ry:+6.1f},{rz:+6.1f}) deg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MyCobot 280 connectivity smoke test")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--skip-home", action="store_true", help="Skip the home move")
    parser.add_argument("--skip-gripper", action="store_true", help="Skip the gripper cycle")
    parser.add_argument(
        "--home-j6",
        type=float,
        default=-45.0,
        help="Joint 6 (gripper yaw) angle at home, in degrees. Default -45 cancels "
             "the +45 deg yaw seen at zero-angle home. Pass 0 for raw home.",
    )
    parser.add_argument(
        "--gripper-close-value",
        type=int,
        default=40,
        help="0=open, 100=fully closed. 40 ≈ shot cup grip.",
    )
    args = parser.parse_args()

    print(f"[smoke] connecting to MyCobot 280 on {args.port} @ {args.baudrate}")
    driver = MyCobotDriver(
        MyCobotDriverSettings(
            port=args.port,
            baudrate=args.baudrate,
            default_speed=args.speed,
            home_angles_deg=(0.0, 0.0, 0.0, 0.0, 0.0, float(args.home_j6)),
        )
    )
    driver.connect()
    try:
        if not driver.is_power_on():
            print("[smoke] power_on")
            driver.power_on()

        coords = driver.get_coords_mm_deg()
        print(f"[smoke] startup pose:  {_fmt_coords(coords)}")
        angles = driver.get_angles_deg()
        print(f"[smoke] startup angles: {[round(a, 1) for a in angles]} deg")

        if not args.skip_home:
            print("[smoke] homing")
            driver.home()
            print(f"[smoke] post-home pose: {_fmt_coords(driver.get_coords_mm_deg())}")

        if not args.skip_gripper:
            gripper = Gripper(driver, GripperSettings())
            print("[smoke] gripper -> open")
            gripper.open()
            time.sleep(0.5)
            print(f"[smoke] gripper -> close (value={args.gripper_close_value})")
            gripper.close(value=int(args.gripper_close_value))
            time.sleep(0.5)
            print("[smoke] gripper -> open")
            gripper.open()
            v = gripper.get_value()
            print(f"[smoke] gripper value after open: {v}")

        print("[smoke] DONE")
    finally:
        driver.disconnect()


if __name__ == "__main__":
    main()
