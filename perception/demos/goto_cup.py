"""Detect a cup and hover the gripper above it, in one shot.

Combines `cup_locator_demo --once` with `goto_world`: run image processing
to find the cup, then send the arm to hover above its world X-Y. No copy-
paste of coordinates between commands.

Usage:
    # Default: hover 70 mm above the table, centered on the cup.
    python -m perception.demos.goto_cup \
        --profile calibration/profiles/session_multitag.json \
        --port /dev/ttyUSB0

    # First-run safety: more clearance.
    python -m perception.demos.goto_cup --hover-mm 100 --speed 10 ...

    # Detection only, no robot motion.
    python -m perception.demos.goto_cup --dry-run ...

Exit codes:
    0  hover completed
    2  no stable cup detected within --timeout-s
    3  hover target out of reach
"""
from __future__ import annotations

import argparse
import sys
import time

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
from perception.cup_locator import CupLocator


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m perception.demos.goto_cup",
        description=(
            "Detect a cup with the camera (cup_locator) and hover the gripper "
            "above its X-Y at the requested height above the table."
        ),
    )
    p.add_argument(
        "--profile",
        default="calibration/profiles/session_multitag.json",
        help="Path to the calibration profile.",
    )
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=1_000_000)

    # --- Detection ----------------------------------------------------------
    p.add_argument("--target-label", default="cup")
    p.add_argument("--min-confidence", type=float, default=0.15)
    p.add_argument("--settle-frames", type=int, default=8)
    p.add_argument(
        "--timeout-s",
        type=float,
        default=4.0,
        help="How long to wait for a stable detection before giving up.",
    )
    p.add_argument("--device-index", type=int, default=0)

    # --- Motion -------------------------------------------------------------
    p.add_argument(
        "--hover-mm",
        type=float,
        default=70.0,
        help="Height above the table plane to hover at (default 70 mm = ~2 cm "
             "above a 5 cm cup's rim).",
    )
    p.add_argument("--speed", type=int, default=15)
    p.add_argument(
        "--rpy",
        type=float,
        nargs=3,
        default=None,
        help="Override vertical RPY (deg). Default: 180 0 0.",
    )
    p.add_argument("--max-reach-mm", type=float, default=270.0)
    p.add_argument(
        "--use-ik", action="store_true",
        help="Use URDF-based IK + send_angles instead of firmware send_coords "
             "(requires a fitted JointMap: ik_debug.py compare --save).",
    )
    p.add_argument(
        "--ik-orient", choices=("none", "Z", "all"), default="Z",
        help="IK orientation constraint (default Z = approach axis).",
    )
    p.add_argument(
        "--no-home-after",
        action="store_true",
        help="Skip the return-to-home at the end.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect the cup and print the planned hover target without moving the arm.",
    )
    args = p.parse_args()

    # --- 1. Detect the cup --------------------------------------------------
    print(f"[goto_cup] detecting {args.target_label!r} (timeout {args.timeout_s:.1f}s)...")
    with CupLocator(
        args.profile,
        target_label=args.target_label,
        min_confidence=args.min_confidence,
        settle_frames=args.settle_frames,
    ) as loc:
        pose = loc.locate(
            timeout_s=args.timeout_s,
            device_index=args.device_index,
        )

    if pose is None:
        print(
            f"[goto_cup] no stable {args.target_label!r} detected within "
            f"{args.timeout_s:.1f}s. Check (a) cup is in camera view, "
            f"(b) ChArUco board visible, (c) profile path correct.",
            file=sys.stderr,
        )
        sys.exit(2)

    wx, wy, wz = pose.position_world_m
    rx, ry, rz = pose.position_robot_mm
    height_str = "n/a" if pose.height_m is None else f"{pose.height_m * 1000:.1f} mm"
    print(
        f"[goto_cup] cup detected:\n"
        f"  world_mm  = ({wx*1000:+8.2f}, {wy*1000:+8.2f}, {wz*1000:+8.2f})\n"
        f"  robot_mm  = ({rx:+8.2f}, {ry:+8.2f}, {rz:+8.2f})\n"
        f"  height    = {height_str}\n"
        f"  quality   = {pose.quality:.2f}   track = {pose.track_id}"
    )

    # --- 2. Build motion context from the same profile ----------------------
    profile = CalibrationProfileIO.load(args.profile)
    t_robot_world = profile.get_robot_world_transform()
    if t_robot_world is None or profile.table_plane is None:
        print(
            f"[goto_cup] profile {args.profile!r} is missing robot_world_transform "
            f"or table_plane; cannot plan motion.",
            file=sys.stderr,
        )
        sys.exit(2)
    n_world, o_world = profile.table_plane.as_arrays()
    rpy_override = (
        tuple(float(v) for v in args.rpy)
        if args.rpy is not None
        else MotionSettings().vertical_rpy_deg
    )
    settings = MotionSettings(
        max_reach_m=float(args.max_reach_mm) * 1e-3,
        default_speed=int(args.speed),
        vertical_rpy_deg=rpy_override,
        use_ik_solver=bool(args.use_ik),
        ik_orientation_mode=str(args.ik_orient),
    )
    ctx = MotionContext(
        t_robot_world=t_robot_world,
        table_normal_world=n_world,
        table_origin_world=o_world,
        settings=settings,
    )

    cup_world_m = np.array([wx, wy, wz], dtype=np.float64)
    hover_target = project_above_table(
        cup_world_m, float(args.hover_mm) * 1e-3, ctx
    )
    reachable, dist = is_reachable(hover_target, ctx)
    p_robot = world_to_robot(hover_target, t_robot_world)
    print(
        f"[goto_cup] hover target:\n"
        f"  world_mm  = ({hover_target[0]*1000:+8.2f}, "
        f"{hover_target[1]*1000:+8.2f}, {hover_target[2]*1000:+8.2f})\n"
        f"  robot_mm  = ({p_robot[0]*1000:+8.2f}, "
        f"{p_robot[1]*1000:+8.2f}, {p_robot[2]*1000:+8.2f})\n"
        f"  reach     = {dist*1000:.1f} mm vs max "
        f"{settings.max_reach_m*1000:.1f} mm  -> "
        f"{'OK' if reachable else 'OUT OF RANGE'}\n"
        f"  RPY       = {settings.vertical_rpy_deg}"
    )

    if args.dry_run:
        print("[goto_cup] --dry-run; not moving.")
        return
    if not reachable:
        print("[goto_cup] ABORT: hover target out of reach.", file=sys.stderr)
        sys.exit(3)

    # --- 3. Connect arm and move --------------------------------------------
    driver = MyCobotDriver(MyCobotDriverSettings(port=args.port, baudrate=args.baudrate))
    driver.connect()
    try:
        driver.power_on()
        time.sleep(0.5)
        gripper = Gripper(driver, GripperSettings())
        gripper.open(wait=False)

        # Pre-pose to a "shoulder forward, elbow down" config to bias the IK
        # solver and avoid silent send_coords rejection on big jumps from home.
        prepose = [0.0, -30.0, -30.0, 0.0, 0.0, -45.0]
        try:
            driver.send_angles_deg(prepose, speed=40)
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                try:
                    cur = driver.get_angles_deg(retries=2)
                except Exception:
                    time.sleep(0.1)
                    continue
                if max(abs(cur[i] - prepose[i]) for i in range(6)) <= 3.0:
                    break
                time.sleep(0.15)
        except Exception as exc:
            print(f"[goto_cup] WARN: pre-pose failed: {exc}")

        try:
            pose_before = driver.get_coords_mm_deg(retries=4)
            print(
                f"[goto_cup] pose BEFORE move (mm/deg): "
                f"({pose_before[0]:+.1f}, {pose_before[1]:+.1f}, {pose_before[2]:+.1f}, "
                f"{pose_before[3]:+.1f}, {pose_before[4]:+.1f}, {pose_before[5]:+.1f})"
            )
        except Exception as exc:
            pose_before = None
            print(f"[goto_cup] could not read pose before move: {exc}")

        try:
            move_to_world(driver, hover_target, ctx, speed=int(args.speed))
        except ReachabilityError as exc:
            print(f"[goto_cup] reach error: {exc}", file=sys.stderr)
            sys.exit(3)

        try:
            pose_after = driver.get_coords_mm_deg(retries=4)
            print(
                f"[goto_cup] pose AFTER  move (mm/deg): "
                f"({pose_after[0]:+.1f}, {pose_after[1]:+.1f}, {pose_after[2]:+.1f}, "
                f"{pose_after[3]:+.1f}, {pose_after[4]:+.1f}, {pose_after[5]:+.1f})"
            )
            if pose_before is not None:
                dx = pose_after[0] - pose_before[0]
                dy = pose_after[1] - pose_before[1]
                dz = pose_after[2] - pose_before[2]
                delta_mm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                print(f"[goto_cup] arm tip moved {delta_mm:.1f} mm "
                      f"(dx={dx:+.1f}, dy={dy:+.1f}, dz={dz:+.1f})")
                if delta_mm < 5.0:
                    print("[goto_cup] WARN: arm barely moved. IK solver may have "
                          "rejected the target. Try a different --rpy.")
        except Exception as exc:
            print(f"[goto_cup] could not read pose after move: {exc}")

        print(
            "[goto_cup] holding pose for 2 s, then "
            + ("exiting." if args.no_home_after else "returning home.")
        )
        time.sleep(2.0)
        if not args.no_home_after:
            driver.home(speed=int(args.speed))
    finally:
        driver.disconnect()


if __name__ == "__main__":
    main()
