"""Interactive 3-point calibration of T_robot_world.

Procedure
---------
You jog the gripper tip (by hand, with servos released) to known world-frame
points one at a time, pressing Enter after each placement. The script reads
the robot's reported tool0 pose, applies the tip offset to estimate the tip
position in the robot frame, then solves the Kabsch alignment to recover
T_robot_world. The transform is stored in the same calibration profile JSON
that perception already uses.

Run from the repo root (NOT inside perception/):

    python3 -m perception.demos.calibrate_robot

Recommended preparation:
    1. Ensure the perception profile is up to date (tags + table plane fit).
    2. Tape a small visible cross at a known offset from tag 1 — this is the
       third calibration point. Measure the offset in mm and write it down
       BEFORE running. You'll be asked to type it.
    3. Power the arm on with a clear workspace.

The script will:
    a. Connect to the arm and release servos so you can hand-move it.
    b. Walk you through points 1 (tag 1 center), 2 (tag 3 center), 3 (taped
       cross). Press 'r' before Enter to redo the previous point.
    c. Print per-point residual; if RMSE < 1.5 cm, write the transform into
       the profile (and into calibration/robot_tag_transform.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from perception.calibration import CalibrationProfileIO
from perception.calibration.robot_calibrator import (
    gripper_tip_position_in_robot,
    kabsch_align,
)
from perception.control import MyCobotDriver, MyCobotDriverSettings


_DEFAULT_PROFILE = "calibration/profiles/session_multitag.json"
_BACKWARDS_COMPAT_FILE = "calibration/robot_tag_transform.json"


def _input_with_redo(prompt: str) -> str:
    """Reads a line; returns 'redo' if user typed 'r', otherwise the raw line."""
    s = input(prompt).strip().lower()
    if s == "r":
        return "redo"
    return s


def _parse_xyz_mm(text: str) -> np.ndarray:
    """Parse 'x y z' in millimeters."""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("expected three numbers separated by space, e.g. '200 100 0'")
    return np.array([float(p) for p in parts], dtype=np.float64) * 1e-3  # mm -> m


def _record_point(
    driver: MyCobotDriver,
    label: str,
    world_xyz_m: np.ndarray,
    tip_offset_z_m: float,
) -> np.ndarray:
    while True:
        print(f"\n--- {label} ---")
        print(f"  world target XYZ: ({world_xyz_m[0]*1000:+.1f}, "
              f"{world_xyz_m[1]*1000:+.1f}, {world_xyz_m[2]*1000:+.1f}) mm")
        print("  Move the gripper tip to that physical location (point straight DOWN).")
        action = _input_with_redo("  Press Enter when in place (or 'r' to re-try this point): ")
        if action == "redo":
            continue
        try:
            coords = driver.get_coords_mm_deg()
        except Exception as exc:
            print(f"  could not read robot coords: {exc} — retrying")
            continue
        tip_xyz_m = gripper_tip_position_in_robot(
            coords_mm_deg=coords,
            tip_offset_z_m=float(tip_offset_z_m),
            assume_pointing_down=True,
        )
        print(
            f"  recorded tool0 pose: xyz=({coords[0]:+.1f},{coords[1]:+.1f},{coords[2]:+.1f}) mm "
            f"rpy=({coords[3]:+.1f},{coords[4]:+.1f},{coords[5]:+.1f}) deg"
        )
        print(
            f"  -> tip in robot frame: ({tip_xyz_m[0]*1000:+.1f}, "
            f"{tip_xyz_m[1]*1000:+.1f}, {tip_xyz_m[2]*1000:+.1f}) mm"
        )
        confirm = input("  Accept this point? [Y/n/r=redo]: ").strip().lower()
        if confirm == "r" or confirm == "n":
            continue
        return tip_xyz_m


def main() -> None:
    parser = argparse.ArgumentParser(description="3-point Kabsch calibration of T_robot_world")
    parser.add_argument("--profile", default=_DEFAULT_PROFILE)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--tip-offset-mm",
        type=float,
        default=95.0,
        help="Gripper tip offset from tool0 flange along tool0-Z, in mm (default 95).",
    )
    parser.add_argument(
        "--no-tip-offset",
        action="store_true",
        help="Disable tip offset (touch the flange, not the gripper tip, to each point).",
    )
    parser.add_argument(
        "--origin-tag-id", type=int, default=1, help="Tag whose center is world (0,0,0)."
    )
    parser.add_argument(
        "--aux-tag-id", type=int, default=3, help="Second tag (used as point 2)."
    )
    parser.add_argument(
        "--rmse-warn-mm",
        type=float,
        default=15.0,
        help="If alignment RMSE exceeds this (mm), print a warning before saving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the transform but do not write it to the profile.",
    )
    args = parser.parse_args()

    tip_offset_m = 0.0 if args.no_tip_offset else float(args.tip_offset_mm) * 1e-3

    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"[calibrate_robot] profile not found: {profile_path}")
        sys.exit(1)
    profile = CalibrationProfileIO.load(str(profile_path))

    p1_world = np.zeros(3, dtype=np.float64)  # origin tag is world (0,0,0) by construction
    if int(args.origin_tag_id) != profile.origin_tag_id:
        print(f"[calibrate_robot] warning: --origin-tag-id={args.origin_tag_id} differs "
              f"from profile.origin_tag_id={profile.origin_tag_id}")

    aux_transform = profile.get_world_tag_transform(int(args.aux_tag_id))
    if aux_transform is None:
        print(f"[calibrate_robot] tag {args.aux_tag_id} not present in profile. "
              "Run auto_calibrate_tags first or change --aux-tag-id.")
        sys.exit(1)
    p2_world = aux_transform[:3, 3]

    print("\n[calibrate_robot] Connecting to arm...")
    driver = MyCobotDriver(
        MyCobotDriverSettings(port=args.port, baudrate=args.baudrate)
    )
    driver.connect()

    try:
        if not driver.is_power_on():
            driver.power_on()

        print("[calibrate_robot] Releasing servos so you can jog the arm by hand.")
        print("                  (Encoders still report position.)")
        driver.release_all_servos()

        print("\n=================================================================")
        print(" 3-point calibration of T_robot_world")
        print(f"   point 1: tag {profile.origin_tag_id} center      -> world (0, 0, 0) mm")
        print(f"   point 2: tag {args.aux_tag_id} center      -> world "
              f"({p2_world[0]*1000:+.1f}, {p2_world[1]*1000:+.1f}, {p2_world[2]*1000:+.1f}) mm")
        print(f"   point 3: a third taped point on the table  -> you'll enter its world XYZ in mm")
        print(f"   tip offset along tool0-Z: {tip_offset_m*1000:.1f} mm "
              f"(assumes gripper points straight DOWN at each touch)")
        print("=================================================================\n")

        tip1 = _record_point(driver, "POINT 1 (tag 1 center)", p1_world, tip_offset_m)
        tip2 = _record_point(driver, f"POINT 2 (tag {args.aux_tag_id} center)", p2_world, tip_offset_m)

        # Third point
        while True:
            text = input(
                "\nEnter the world-frame XYZ of POINT 3 in millimeters "
                "(e.g. '200 100 0' for (0.20, 0.10, 0.0) m): "
            )
            try:
                p3_world = _parse_xyz_mm(text)
                break
            except ValueError as e:
                print(f"  invalid input: {e}")
        tip3 = _record_point(driver, "POINT 3 (taped marker)", p3_world, tip_offset_m)

        world_points = np.stack([p1_world, p2_world, p3_world], axis=0)
        robot_points = np.stack([tip1, tip2, tip3], axis=0)

        # Check collinearity — 3 collinear points make the rotation under-constrained.
        v1 = world_points[1] - world_points[0]
        v2 = world_points[2] - world_points[0]
        cross_norm = float(np.linalg.norm(np.cross(v1, v2)))
        if cross_norm < 1e-4:
            print("\n[calibrate_robot] ERROR: the three world points are (near-)collinear. "
                  "Pick a third point that's clearly off the tag1-tag3 line.")
            sys.exit(2)

        result = kabsch_align(world_points, robot_points)
        print("\n--- Kabsch result ---")
        print(f"  per-point residuals (mm): "
              f"{[round(float(r)*1000, 2) for r in result.per_point_residual_m]}")
        print(f"  RMSE: {result.rmse_m*1000:.2f} mm")
        print(f"  T_robot_world =\n{result.t_robot_world}")

        if result.rmse_m * 1000 > float(args.rmse_warn_mm):
            print(f"\n[calibrate_robot] WARNING: RMSE > {args.rmse_warn_mm:.1f} mm. "
                  "Likely causes:")
            print("  - tip offset is wrong (try --tip-offset-mm 80 ... 110)")
            print("  - gripper wasn't pointing straight down during touches")
            print("  - point 3 is too close to the tag1-tag3 line")
            print("  - manual placement was sloppy on one of the touches")
            confirm = input("  Save anyway? [y/N]: ").strip().lower()
            if confirm != "y":
                print("[calibrate_robot] aborted; profile unchanged.")
                return

        if args.dry_run:
            print("\n[calibrate_robot] --dry-run set; not writing to disk.")
            return

        profile.set_robot_world_transform(result.t_robot_world)
        profile.metrics["robot_world_kabsch_rmse_m"] = float(result.rmse_m)
        CalibrationProfileIO.save(profile, str(profile_path))

        # Also write a flat copy for backward-compat with the old placeholder file
        import json
        Path(_BACKWARDS_COMPAT_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(_BACKWARDS_COMPAT_FILE).write_text(
            json.dumps(
                {
                    "description": "Transform from world (tag origin) to robot base.",
                    "units": "meters",
                    "convention": "p_robot = T_robot_world * p_world",
                    "T_robot_world": result.t_robot_world.tolist(),
                    "rmse_m": float(result.rmse_m),
                },
                indent=2,
            )
        )
        print(f"\n[calibrate_robot] saved to {profile_path}")
        print(f"[calibrate_robot] also wrote {_BACKWARDS_COMPAT_FILE}")

    finally:
        try:
            driver.release_all_servos()
        except Exception:
            pass
        driver.disconnect()


if __name__ == "__main__":
    main()
