"""End-to-end sanity check for the cup_locator pipeline.

Run this after touch-calibration to confirm the saved profile produces sensible
cup coordinates in the MyCobot base frame. The demo polls `CupLocator.locate()`
in a loop and prints each result; move the cup around and watch the numbers
track. No robot connection is required — only the Record3D camera and a printed
ChArUco board visible to it.

Usage:
    # Default: continuous polling until Ctrl+C.
    python -m perception.demos.cup_locator_demo \
        --profile calibration/profiles/session_multitag.json

    # One detection then exit (for scripts / CI).
    python -m perception.demos.cup_locator_demo --once

    # Stop after N successful detections.
    python -m perception.demos.cup_locator_demo --max-detections 10

Interpretation tips:
- Watching the numbers stabilise over several polls confirms the temporal
  smoothing (YOLO anchor EMA + world tracker median height + position EMA)
  is doing its job.
- Picking up the cup and moving it 10 cm in one direction should shift the
  reported robot-mm by ~100 mm in a consistent axis; if it moves the wrong
  way or the wrong amount, the calibration is suspect.
- Compare `world_m` against where you eyeballed the cup on the board: the
  world frame's origin is at the ChArUco TL corner, +X along the columns,
  +Y along the rows.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from types import FrameType
from typing import Optional

from perception.cup_locator import CupLocator, CupPose


def _format_pose(pose: CupPose, dt_s: float) -> str:
    x_mm, y_mm, z_mm = pose.position_robot_mm
    wx_m, wy_m, wz_m = pose.position_world_m
    h_mm = "  n/a" if pose.height_m is None else f"{pose.height_m * 1000:5.1f}"
    yaw = "  n/a" if pose.yaw_hint_rad is None else f"{pose.yaw_hint_rad:+5.2f}"
    return (
        f"robot=({x_mm:+7.1f}, {y_mm:+7.1f}, {z_mm:+7.1f}) mm   "
        f"world=({wx_m * 1000:+7.1f}, {wy_m * 1000:+7.1f}, {wz_m * 1000:+7.1f}) mm   "
        f"h={h_mm} mm   q={pose.quality:.2f}   yaw={yaw} rad   "
        f"track={pose.track_id}   dt={dt_s * 1000:5.0f}ms"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m perception.demos.cup_locator_demo",
        description=(
            "Stream cup positions from `CupLocator` after touch-calibration. "
            "Use this to confirm the saved profile produces sane robot-frame "
            "coordinates when you move the cup around the ChArUco board."
        ),
    )
    p.add_argument(
        "--profile",
        default="calibration/profiles/session_multitag.json",
        help="Profile path written by `python -m perception.cup_locator.calibrate`.",
    )
    p.add_argument("--target-label", default="cup")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        help="YOLO confidence floor (lowered from the 0.30 default for reliability).",
    )
    p.add_argument(
        "--settle-frames",
        type=int,
        default=8,
        help="How many consecutive tracker hits before locate() returns a pose.",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=2.0,
        help="Per-poll timeout. Returns None if no stable detection within this window.",
    )
    p.add_argument(
        "--max-detections",
        type=int,
        default=0,
        help="Stop after N successful detections (0 = run until Ctrl+C).",
    )
    p.add_argument(
        "--max-misses",
        type=int,
        default=0,
        help="Stop after N consecutive timeouts (0 = never give up).",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Equivalent to --max-detections 1.",
    )
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args()

    stop = {"flag": False}

    def _on_sigint(_signum: int, _frame: Optional[FrameType]) -> None:
        if stop["flag"]:
            # Second Ctrl+C — bail hard.
            print("\n[cup_locator_demo] second interrupt; exiting immediately.")
            sys.exit(130)
        stop["flag"] = True
        print(
            "\n[cup_locator_demo] interrupt received; will exit after current "
            "locate() returns. Press Ctrl+C again to force-quit.",
            flush=True,
        )

    signal.signal(signal.SIGINT, _on_sigint)

    target_after = (
        1 if args.once
        else (args.max_detections if args.max_detections > 0 else None)
    )

    print(f"[cup_locator_demo] profile        = {args.profile}")
    print(f"[cup_locator_demo] target_label   = {args.target_label!r}")
    print(f"[cup_locator_demo] min_confidence = {args.min_confidence}")
    print(f"[cup_locator_demo] settle_frames  = {args.settle_frames}")
    print(f"[cup_locator_demo] timeout_s      = {args.timeout_s}")
    if target_after is not None:
        print(f"[cup_locator_demo] stop after     = {target_after} detection(s)")
    else:
        print("[cup_locator_demo] stop after     = Ctrl+C")
    print()

    successes = 0
    misses = 0
    consecutive_misses = 0
    iteration = 0

    with CupLocator(
        args.profile,
        target_label=args.target_label,
        min_confidence=args.min_confidence,
        settle_frames=args.settle_frames,
    ) as loc:
        print("[cup_locator_demo] camera opens on first locate() call. Polling...")
        while not stop["flag"]:
            iteration += 1
            t0 = time.monotonic()
            pose = loc.locate(
                timeout_s=args.timeout_s,
                device_index=args.device_index,
            )
            dt = time.monotonic() - t0
            if pose is None:
                misses += 1
                consecutive_misses += 1
                print(
                    f"[{iteration:3d}] no '{args.target_label}' detected   "
                    f"({dt:.2f}s)   total_misses={misses}",
                    flush=True,
                )
                if args.max_misses > 0 and consecutive_misses >= args.max_misses:
                    print(
                        f"[cup_locator_demo] hit --max-misses={args.max_misses}; "
                        "giving up."
                    )
                    break
            else:
                consecutive_misses = 0
                successes += 1
                print(f"[{iteration:3d}] {_format_pose(pose, dt)}", flush=True)
                if target_after is not None and successes >= target_after:
                    break

    print()
    print(
        f"[cup_locator_demo] done. iterations={iteration} "
        f"successes={successes} misses={misses}"
    )
    if successes == 0:
        print(
            "[cup_locator_demo] no successful detections — verify (a) a cup is "
            "in view, (b) the ChArUco board is visible to the camera, and "
            "(c) the profile path is correct.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
