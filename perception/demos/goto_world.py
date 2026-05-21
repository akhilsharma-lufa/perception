"""Move the arm to a world-frame XYZ point (sanity-check for the calibration).

Run from the repo root, e.g.:

    # Just print what would be sent — no motion:
    python3 -m perception.demos.goto_world --world-mm -64 121 -46 --dry-run

    # Hover 8 cm above the (X, Y) of that world point, regardless of the
    # given Z; useful for "hover above the cup":
    python3 -m perception.demos.goto_world --world-mm -64 121 -46 --hover-mm 80

    # Touch the world point directly:
    python3 -m perception.demos.goto_world --world-mm -64 121 -46

    # Override the "vertical" RPY if the gripper points the wrong way:
    python3 -m perception.demos.goto_world --world-mm 0 0 0 --hover-mm 80 --rpy 180 0 -90

    # Lower the speed for safety on first runs:
    python3 -m perception.demos.goto_world --world-mm 0 0 0 --hover-mm 80 --speed 15

Workflow:
  1. With the red cup placed, read its world coords from yolo_world_monitor.
  2. Call this demo with --hover-mm 80 to position the gripper ~8 cm above
     the cup. Verify the gripper is over the cup with the tip pointing down.
  3. If the gripper points the wrong direction, retry with --rpy. Once you
     find an RPY that gives a vertical-down gripper, the same value will
     work for the upcoming pick FSM.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from perception.calibration import CalibrationProfileIO
from perception.control import (
    Gripper,
    GripperSettings,
    MotionContext,
    MotionSettings,
    MyCobotDriver,
    MyCobotDriverSettings,
    ReachabilityError,
    is_reachable,
    move_to_world,
    project_above_table,
    world_to_robot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Move arm to a world-frame XYZ point.")
    parser.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--world-mm",
        type=float,
        nargs=3,
        required=True,
        help="Target world XYZ in mm, e.g. --world-mm -64 121 -46",
    )
    parser.add_argument(
        "--hover-mm",
        type=float,
        default=0.0,
        help="If > 0, project the target to this height above the table plane "
             "(ignores the given Z). Use for 'hover above the cup'.",
    )
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument(
        "--rpy",
        type=float,
        nargs=3,
        default=None,
        help="Override vertical RPY (deg). Default: 180 0 0.",
    )
    parser.add_argument(
        "--max-reach-mm",
        type=float,
        default=270.0,
        help="Conservative reach gate (default 270 mm).",
    )
    parser.add_argument(
        "--coord-mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="pymycobot send_coords mode: 0=angular interp (default, robust), "
             "1=linear cartesian interp (fails silently if path isn't reachable).",
    )
    parser.add_argument(
        "--no-home-after",
        action="store_true",
        help="Skip the return-to-home at the end (default homes the arm).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute only, don't move.")
    args = parser.parse_args()

    profile = CalibrationProfileIO.load(args.profile)
    if profile.robot_world_transform is None:
        print(f"[goto_world] ERROR: profile has no robot_world_transform. "
              f"Run calibrate_robot first.")
        sys.exit(2)
    if profile.table_plane is None:
        print(f"[goto_world] ERROR: profile has no table_plane. "
              f"Run auto_calibrate_tags first.")
        sys.exit(2)

    t_robot_world = profile.get_robot_world_transform()
    n_world, o_world = profile.table_plane.as_arrays()

    settings = MotionSettings(
        max_reach_m=float(args.max_reach_mm) * 1e-3,
        default_speed=int(args.speed),
        vertical_rpy_deg=(
            tuple(float(v) for v in args.rpy)
            if args.rpy is not None
            else MotionSettings().vertical_rpy_deg
        ),
        coord_mode=int(args.coord_mode),
    )
    ctx = MotionContext(
        t_robot_world=t_robot_world,
        table_normal_world=n_world,
        table_origin_world=o_world,
        settings=settings,
    )

    p_world_m = np.asarray(args.world_mm, dtype=np.float64) * 1e-3
    if args.hover_mm > 0:
        target = project_above_table(p_world_m, float(args.hover_mm) * 1e-3, ctx)
    else:
        target = p_world_m

    reachable, dist = is_reachable(target, ctx)
    p_robot = world_to_robot(target, t_robot_world)
    print(f"[goto_world] target world (mm): "
          f"({target[0]*1000:+.1f}, {target[1]*1000:+.1f}, {target[2]*1000:+.1f})")
    print(f"[goto_world] target robot (mm): "
          f"({p_robot[0]*1000:+.1f}, {p_robot[1]*1000:+.1f}, {p_robot[2]*1000:+.1f})")
    print(f"[goto_world] reach: {dist*1000:.1f} mm "
          f"vs max {settings.max_reach_m*1000:.1f} mm -> "
          f"{'OK' if reachable else 'OUT OF RANGE'}")
    print(f"[goto_world] vertical RPY (deg): {settings.vertical_rpy_deg}")

    if args.dry_run:
        print("[goto_world] --dry-run; not moving.")
        return

    if not reachable:
        print("[goto_world] ABORT: target out of reach.")
        sys.exit(3)

    driver = MyCobotDriver(MyCobotDriverSettings(port=args.port, baudrate=args.baudrate))
    driver.connect()
    try:
        if not driver.is_power_on():
            driver.power_on()
        gripper = Gripper(driver, GripperSettings())
        gripper.open(wait=False)
        # Snapshot pose BEFORE the move so we can tell if the arm actually went anywhere.
        try:
            pose_before = driver.get_coords_mm_deg(retries=4)
            print(f"[goto_world] pose BEFORE move (mm/deg): "
                  f"({pose_before[0]:+.1f}, {pose_before[1]:+.1f}, {pose_before[2]:+.1f}, "
                  f"{pose_before[3]:+.1f}, {pose_before[4]:+.1f}, {pose_before[5]:+.1f})")
        except Exception as exc:
            pose_before = None
            print(f"[goto_world] could not read pose before move: {exc}")
        try:
            move_to_world(driver, target, ctx, speed=int(args.speed))
        except ReachabilityError as e:
            print(f"[goto_world] reach error: {e}")
            sys.exit(3)
        try:
            pose_after = driver.get_coords_mm_deg(retries=4)
            print(f"[goto_world] pose AFTER  move (mm/deg): "
                  f"({pose_after[0]:+.1f}, {pose_after[1]:+.1f}, {pose_after[2]:+.1f}, "
                  f"{pose_after[3]:+.1f}, {pose_after[4]:+.1f}, {pose_after[5]:+.1f})")
            if pose_before is not None:
                dx = pose_after[0] - pose_before[0]
                dy = pose_after[1] - pose_before[1]
                dz = pose_after[2] - pose_before[2]
                delta_mm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                print(f"[goto_world] arm tip moved {delta_mm:.1f} mm "
                      f"(dx={dx:+.1f}, dy={dy:+.1f}, dz={dz:+.1f})")
                if delta_mm < 5.0:
                    print(f"[goto_world] WARN: arm barely moved. IK solver may have rejected the target "
                          f"(XYZ+RPY combination has no valid solution). Try a different --rpy.")
            tx_mm = p_robot[0] * 1000.0
            ty_mm = p_robot[1] * 1000.0
            tz_mm = p_robot[2] * 1000.0
            err_xyz = float(np.sqrt(
                (pose_after[0] - tx_mm) ** 2
                + (pose_after[1] - ty_mm) ** 2
                + (pose_after[2] - tz_mm) ** 2
            ))
            print(f"[goto_world] error vs commanded target: {err_xyz:.1f} mm")
        except Exception as exc:
            print(f"[goto_world] could not read pose after move: {exc}")
        print("[goto_world] holding pose for 2 s, then "
              + ("returning home." if not args.no_home_after else "releasing servos."))
        import time as _t
        _t.sleep(2.0)
        if not args.no_home_after:
            driver.home(speed=int(args.speed))
    finally:
        try:
            driver.release_all_servos()
        except Exception:
            pass
        driver.disconnect()


if __name__ == "__main__":
    main()
