"""Manual touch-to-corner robot calibration.

You free-drive the arm by hand so the pointer tip touches a known ChArUco
board corner; the script records the robot's reported pose and pairs it with
the corner's known world position. Three or more non-collinear touches give
a complete Kabsch fit for `T_robot_world` — typically far more accurate than
the depth-peak auto-sweep, because the world position of each touch is
mathematically exact (set by the printed board geometry) and the robot pose
is read directly from the encoders.

Targets you can name:
- Outer paper-edge corners: `TL`, `TR`, `BL`, `BR`. Easiest to land a pointer
  on; world position derived from board dimensions.
- Inner chessboard corners: `col,row` (0-indexed). Many to choose from but
  visually harder to pick out unless you compare against an annotated board.

Commands inside the prompt loop:
  free                  — release servos so you can move the arm by hand
  lock                  — re-engage servos at the current pose
  TL|TR|BL|BR           — record an outer-corner touch
  col,row               — record an inner-corner touch (e.g. "5,3")
  show                  — print current robot pose + last board detection
  list                  — list recorded samples so far
  drop N                — remove sample N from the list
  fit                   — run Kabsch, print residuals
  save                  — save the fit into the calibration profile
  quit                  — exit

Setup:
  - ChArUco board flat on the table, fully in the iPhone's view.
  - Pointer attached to the end face (12 mm protrusion along tool-Z).
  - iPhone in the tripod, Record3D streaming to the Jetson.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    detect_board_pose,
)
from perception.calibration.profiles import (
    CalibrationProfile,
    CalibrationProfileIO,
    CharucoBoardSpec,
)
from perception.calibration.robot_calibrator import (
    gripper_tip_position_in_robot,
    kabsch_align,
)
from perception.control.mycobot_driver import MyCobotDriver, MyCobotDriverSettings
from perception.io import Record3DSource


def _outer_corners_m(cfg: CharucoBoardConfig) -> Dict[str, np.ndarray]:
    """Return the four outer paper-edge corner world positions.

    OpenCV's CharucoBoard convention puts the board frame origin at the outer
    corner of square (0, 0), with X along the columns and Y along the rows
    (Z out of the board surface). The outer corners of the entire board are
    therefore at multiples of squares_x * square and squares_y * square.
    """
    sx = float(cfg.squares_x) * float(cfg.square_length_m)
    sy = float(cfg.squares_y) * float(cfg.square_length_m)
    return {
        "TL": np.array([0.0, 0.0, 0.0]),
        "TR": np.array([sx, 0.0, 0.0]),
        "BL": np.array([0.0, sy, 0.0]),
        "BR": np.array([sx, sy, 0.0]),
    }


def _inner_corner_m(col: int, row: int, cfg: CharucoBoardConfig) -> np.ndarray:
    """Position of the inner chessboard corner at index (col, row)."""
    if not (0 <= col <= cfg.squares_x - 2):
        raise ValueError(f"col {col} out of range [0, {cfg.squares_x - 2}]")
    if not (0 <= row <= cfg.squares_y - 2):
        raise ValueError(f"row {row} out of range [0, {cfg.squares_y - 2}]")
    s = float(cfg.square_length_m)
    return np.array([(col + 1) * s, (row + 1) * s, 0.0])


def _enable_free_drive(driver: MyCobotDriver, on: bool) -> None:
    """Toggle the arm between hand-movable and locked at current pose.

    Strategy:
      - On hand-movable: release_all_servos() — gravity will pull the arm down
        slightly, so the user should support the arm while moving it.
      - On locked: read current angles, send them back as a target — this
        re-engages the servos at the current pose so the arm holds still.
    """
    mc = driver._require_connected()
    if on:
        try:
            driver.release_all_servos()
        except Exception as exc:
            print(f"  warn: release_all_servos failed: {exc}")
        return
    # Re-engage at the current angles so the arm freezes where the user left it.
    try:
        cur = driver.get_angles_deg(retries=4)
    except Exception as exc:
        print(f"  warn: could not read angles to re-engage: {exc}")
        return
    try:
        driver.send_angles_deg(cur, speed=10)
    except Exception as exc:
        print(f"  warn: send_angles_deg failed: {exc}")


def _parse_target(
    text: str, cfg: CharucoBoardConfig, outer: Dict[str, np.ndarray]
) -> Optional[Tuple[str, np.ndarray]]:
    s = text.strip().upper()
    if s in outer:
        return s, outer[s]
    # Try "col,row" or "col row"
    parts = s.replace(",", " ").split()
    if len(parts) == 2:
        try:
            col, row = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        try:
            pos = _inner_corner_m(col, row, cfg)
        except ValueError as exc:
            print(f"  {exc}")
            return None
        return f"inner({col},{row})", pos
    return None


def _verify_board(
    source: Record3DSource, cfg: CharucoBoardConfig, secs: float = 3.0
) -> bool:
    print(f"[touch-cal] verifying ChArUco detection ({secs:.0f}s)...")
    deadline = time.monotonic() + float(secs)
    n_ok = 0
    last_print = 0.0
    while time.monotonic() < deadline:
        pkt = source.wait_for_frame(timeout_s=0.25)
        if pkt is None:
            continue
        det = detect_board_pose(pkt.rgb, pkt.intrinsic_mat, cfg)
        if det is None:
            continue
        n_ok += 1
        now = time.monotonic()
        if now - last_print > 0.5:
            tx, ty, tz = det.t_camera_board[:3, 3]
            print(
                f"[touch-cal]   board: t=({tx*1000:+6.1f},{ty*1000:+6.1f},{tz*1000:+6.1f}) mm  "
                f"corners={det.n_corners}  reproj={det.reprojection_error_px:.2f} px"
            )
            last_print = now
    if n_ok == 0:
        print(
            "[touch-cal] FATAL: no board detections. "
            "Check --squares-x/y, --square-mm, --dict, --legacy-pattern."
        )
        return False
    print(f"[touch-cal] OK: {n_ok} detection(s).")
    return True


def _print_targets(cfg: CharucoBoardConfig, outer: Dict[str, np.ndarray]) -> None:
    print()
    print("[touch-cal] AVAILABLE TARGETS")
    print("  Outer corners (paper-edge):")
    for name in ("TL", "TR", "BL", "BR"):
        p = outer[name] * 1000.0
        print(f"    {name}  world = ({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:6.1f}) mm")
    n_in_x = cfg.squares_x - 1
    n_in_y = cfg.squares_y - 1
    s_mm = cfg.square_length_m * 1000.0
    print(
        f"  Inner corners: col in [0..{n_in_x-1}], row in [0..{n_in_y-1}], "
        f"world = ((col+1)*{s_mm:.1f}, (row+1)*{s_mm:.1f}, 0) mm"
    )
    print()


def _fit_and_print(samples: List[Tuple[str, np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    if len(samples) < 3:
        print(
            f"[touch-cal] need at least 3 non-collinear samples to fit Kabsch; "
            f"have {len(samples)}."
        )
        return None
    world = np.stack([s[1] for s in samples], axis=0)
    robot = np.stack([s[2] for s in samples], axis=0)
    result = kabsch_align(world, robot)
    print(f"[touch-cal] Kabsch RMSE = {result.rmse_m*1000:.2f} mm  ({len(samples)} samples)")
    for i, (name, w, r) in enumerate(samples):
        print(
            f"[touch-cal]   {i:2d} {name:>14s}  residual = "
            f"{result.per_point_residual_m[i]*1000:6.2f} mm"
        )
    print("[touch-cal] T_robot_world =")
    with np.printoptions(precision=4, suppress=True):
        print(result.t_robot_world)
    return result.t_robot_world


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=1_000_000)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument(
        "--profile",
        default="calibration/profiles/session_multitag.json",
    )
    # Board geometry
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=20.0)
    p.add_argument("--marker-mm", type=float, default=14.0)
    p.add_argument("--dict", default="DICT_4X4_50")
    p.add_argument(
        "--legacy-pattern",
        dest="legacy_pattern",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-legacy-pattern",
        dest="legacy_pattern",
        action="store_false",
    )
    # Tip
    p.add_argument("--tip-offset-mm", type=float, default=12.0)
    # Flow
    p.add_argument(
        "--start-locked",
        action="store_true",
        default=False,
        help="Start with servos engaged. Default: start in free-drive (released).",
    )
    p.add_argument(
        "--verify-s",
        type=float,
        default=3.0,
        help="Seconds to verify ChArUco detection at startup.",
    )
    args = p.parse_args()

    board_cfg = CharucoBoardConfig(
        squares_x=int(args.squares_x),
        squares_y=int(args.squares_y),
        square_length_m=float(args.square_mm) * 1e-3,
        marker_length_m=float(args.marker_mm) * 1e-3,
        dictionary_name=str(args.dict),
        legacy_pattern=bool(args.legacy_pattern),
    )
    outer = _outer_corners_m(board_cfg)
    tip_offset_z_m = float(args.tip_offset_mm) * 1e-3

    # --- Camera ---------------------------------------------------------
    source = Record3DSource()
    print(f"[touch-cal] connecting to Record3D device #{args.device_index}...")
    source.connect(device_index=int(args.device_index))
    print(
        f"[touch-cal] board: {board_cfg.squares_x}x{board_cfg.squares_y} squares, "
        f"square={board_cfg.square_length_m*1000:.1f}mm, "
        f"marker={board_cfg.marker_length_m*1000:.1f}mm, "
        f"dict={board_cfg.dictionary_name}, "
        f"legacy_pattern={board_cfg.legacy_pattern}"
    )
    if not _verify_board(source, board_cfg, secs=float(args.verify_s)):
        source.disconnect()
        sys.exit(1)

    # --- Arm ------------------------------------------------------------
    driver = MyCobotDriver(MyCobotDriverSettings(
        port=str(args.port), baudrate=int(args.baudrate)
    ))
    print(f"[touch-cal] connecting to MyCobot on {args.port}...")
    driver.connect()
    try:
        driver.power_on()
    except Exception as exc:
        print(f"[touch-cal] power_on warn: {exc}")

    _print_targets(board_cfg, outer)

    if not args.start_locked:
        print(
            "[touch-cal] releasing servos for free-drive. Support the arm with "
            "your hand to keep it from drooping under gravity. Type 'lock' once "
            "the tip is on the target."
        )
        _enable_free_drive(driver, on=True)
        locked = False
    else:
        print("[touch-cal] arm is locked at current pose. Type 'free' to move it.")
        locked = True

    samples: List[Tuple[str, np.ndarray, np.ndarray]] = []
    last_t_robot_world: Optional[np.ndarray] = None

    print()
    print(
        "[touch-cal] enter targets to record (TL/TR/BL/BR or 'col,row'). "
        "Commands: free, lock, show, list, drop N, fit, save, quit"
    )
    try:
        while True:
            try:
                line = input("[touch-cal] > ").strip()
            except EOFError:
                line = "quit"
            if not line:
                continue
            cmd = line.lower()

            if cmd in ("q", "quit", "exit"):
                break

            if cmd == "free":
                _enable_free_drive(driver, on=True)
                locked = False
                print("  servos released. support the arm by hand.")
                continue

            if cmd == "lock":
                _enable_free_drive(driver, on=False)
                locked = True
                print("  arm locked at current pose.")
                continue

            if cmd == "show":
                try:
                    a = driver.get_angles_deg(retries=2)
                    c = driver.get_coords_mm_deg(retries=2)
                    tip = gripper_tip_position_in_robot(c, tip_offset_z_m=tip_offset_z_m)
                    print(
                        f"  angles=[{', '.join(f'{x:6.1f}' for x in a)}]\n"
                        f"  coords=[{', '.join(f'{x:6.1f}' for x in c)}]  (mm/deg)\n"
                        f"  tip_robot=({tip[0]*1000:+7.1f}, {tip[1]*1000:+7.1f}, "
                        f"{tip[2]*1000:+7.1f}) mm  locked={locked}"
                    )
                except Exception as exc:
                    print(f"  warn: {exc}")
                continue

            if cmd == "list":
                if not samples:
                    print("  no samples recorded.")
                else:
                    for i, (name, w, r) in enumerate(samples):
                        print(
                            f"  {i:2d} {name:>14s}  world=({w[0]*1000:+6.1f},"
                            f"{w[1]*1000:+6.1f},{w[2]*1000:+6.1f}) mm  "
                            f"tip_robot=({r[0]*1000:+6.1f},{r[1]*1000:+6.1f},"
                            f"{r[2]*1000:+6.1f}) mm"
                        )
                continue

            if cmd.startswith("drop"):
                try:
                    idx = int(cmd.split()[1])
                    if 0 <= idx < len(samples):
                        removed = samples.pop(idx)
                        print(f"  dropped sample {idx} ({removed[0]}). {len(samples)} remain.")
                    else:
                        print(f"  index out of range; have {len(samples)} samples.")
                except (IndexError, ValueError):
                    print("  usage: drop N")
                continue

            if cmd == "fit":
                last_t_robot_world = _fit_and_print(samples)
                continue

            if cmd == "save":
                if last_t_robot_world is None:
                    print("  nothing to save; run 'fit' first.")
                    continue
                try:
                    profile = CalibrationProfileIO.load(args.profile)
                    print(f"  loaded existing profile: {args.profile}")
                except FileNotFoundError:
                    profile = CalibrationProfile.new_charuco(
                        CharucoBoardSpec(
                            squares_x=board_cfg.squares_x,
                            squares_y=board_cfg.squares_y,
                            square_length_m=board_cfg.square_length_m,
                            marker_length_m=board_cfg.marker_length_m,
                            dictionary_name=board_cfg.dictionary_name,
                            legacy_pattern=board_cfg.legacy_pattern,
                        )
                    )
                    print(f"  created new profile (no prior file at {args.profile})")

                profile.set_robot_world_transform(last_t_robot_world)
                profile.set_table_plane(
                    normal_world=np.array([0.0, 0.0, 1.0]),
                    origin_world=np.array([0.0, 0.0, 0.0]),
                    inlier_ratio=1.0,
                    mean_abs_residual_m=0.0,
                )
                profile.set_charuco_board(
                    CharucoBoardSpec(
                        squares_x=board_cfg.squares_x,
                        squares_y=board_cfg.squares_y,
                        square_length_m=board_cfg.square_length_m,
                        marker_length_m=board_cfg.marker_length_m,
                        dictionary_name=board_cfg.dictionary_name,
                        legacy_pattern=board_cfg.legacy_pattern,
                    )
                )
                profile.metrics = {
                    "robot_world_n_samples": float(len(samples)),
                    "tip_offset_z_m": float(tip_offset_z_m),
                    "calibration_method": 1.0,  # 1 = touch, 0 = depth-peak
                }
                profile.created_at_utc = datetime.now(timezone.utc).isoformat()
                CalibrationProfileIO.save(profile, args.profile)
                print(f"  saved -> {args.profile}")
                continue

            # --- Else: treat as a target name ---------------------------
            parsed = _parse_target(line, board_cfg, outer)
            if parsed is None:
                print(
                    f"  unknown command '{line}'. Expected: TL/TR/BL/BR, 'col,row', "
                    "free, lock, show, list, drop N, fit, save, quit."
                )
                continue
            name, world_pos = parsed

            # Snap servos closed so we read a stable pose. If user said 'free'
            # earlier the arm may have drifted under gravity while typing.
            was_locked = locked
            if not locked:
                _enable_free_drive(driver, on=False)
                locked = True
                time.sleep(0.1)  # let pose settle after re-engage
                print("  (auto-locked to capture stable pose)")
            try:
                coords = driver.get_coords_mm_deg(retries=4)
            except Exception as exc:
                print(f"  failed to read pose: {exc}")
                continue
            tip_robot = gripper_tip_position_in_robot(
                coords, tip_offset_z_m=tip_offset_z_m
            )
            samples.append((name, world_pos.copy(), tip_robot.copy()))
            print(
                f"  recorded {name}: "
                f"world=({world_pos[0]*1000:+6.1f},{world_pos[1]*1000:+6.1f},"
                f"{world_pos[2]*1000:+6.1f}) mm  "
                f"tip_robot=({tip_robot[0]*1000:+6.1f},{tip_robot[1]*1000:+6.1f},"
                f"{tip_robot[2]*1000:+6.1f}) mm. "
                f"({len(samples)} sample(s) total)"
            )
            if not was_locked:
                print("  type 'free' to reposition the arm for the next touch.")

    finally:
        print("[touch-cal] locking servos before exit (so the arm doesn't drop).")
        try:
            _enable_free_drive(driver, on=False)
        except Exception:
            pass
        try:
            driver.disconnect()
        finally:
            source.disconnect()


if __name__ == "__main__":
    main()
