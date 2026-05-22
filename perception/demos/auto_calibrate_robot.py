"""Fully automated robot-world calibration using a ChArUco board.

Setup (one-time, physical):
    - Print a ChArUco board on letter paper. Defaults match: 7x10 inner squares
      of 20 mm with 15 mm ArUco markers from DICT_4X4_50. Tape the four corners
      flat to the table so the paper does not warp.
    - Remove the gripper. Install a single black pointer protruding 12 mm out of
      the center of the end face along tool-Z.
    - iPhone 16 Pro on the tripod, Record3D streaming over USB to the Jetson.

The board defines the world frame at runtime (it replaces the AprilTag flow).
Place the board anywhere in the camera's view; cups can sit anywhere else in
the scene, the board does NOT bound the workspace.

Run on the Jetson:
    python3 -m perception.demos.auto_calibrate_robot --port /dev/ttyUSB1

Useful flags:
    --dry-run           : do not move the arm; print live board detections.
    --tip-offset-mm 12  : pointer length protruding past the end face.
    --square-mm 20      : ChArUco square size (must match the printed sheet).
    --max-rmse-mm 5     : if the fit is worse than this, ask before saving.
    --release-servos    : let the arm fall limp after calibration. Default OFF.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np

from perception.calibration.profiles import (
    CalibrationProfile,
    CalibrationProfileIO,
    CharucoBoardSpec,
)
from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    detect_board_pose,
)
from perception.calibration.tip_detector import TipDetectorSettings
from perception.calibration.auto_robot_calibrator import (
    AutoCalibratorSettings,
    capture_reference_depth,
    collect_samples,
    fit_robot_world,
)
from perception.control.mycobot_driver import (
    MyCobotDriver,
    MyCobotDriverSettings,
)
from perception.io import Record3DSource


def _print_pose_block(label: str, t_camera_board: np.ndarray, det) -> None:
    tx, ty, tz = t_camera_board[:3, 3]
    print(
        f"[auto-cal]   {label}: t=({tx*1000:+6.1f}, {ty*1000:+6.1f}, {tz*1000:+6.1f}) mm  "
        f"corners={det.n_corners}  reproj={det.reprojection_error_px:.2f} px"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Automated ChArUco-based robot-world calibration."
    )
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--profile",
        default="calibration/profiles/session_multitag.json",
        help="Path to load/save the CalibrationProfile JSON.",
    )
    # Board geometry
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=10)
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--marker-mm", type=float, default=15.0)
    parser.add_argument("--dict", default="DICT_4X4_50")
    parser.add_argument(
        "--legacy-pattern",
        dest="legacy_pattern",
        action="store_true",
        default=True,
        help=(
            "Use the legacy ChArUco marker placement (top-left square is a "
            "TAG, not black). This is the convention used by most pre-printed "
            "boards and by OpenCV <= 4.6. Default ON."
        ),
    )
    parser.add_argument(
        "--no-legacy-pattern",
        dest="legacy_pattern",
        action="store_false",
        help="Use the OpenCV 4.7+ default layout (top-left square is BLACK).",
    )
    # Tip & sampling
    parser.add_argument("--tip-offset-mm", type=float, default=12.0)
    parser.add_argument("--frames-per-waypoint", type=int, default=10)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--move-speed", type=int, default=40)
    parser.add_argument("--max-rmse-mm", type=float, default=5.0)
    # Workspace orientation
    parser.add_argument(
        "--j1-center",
        type=float,
        default=90.0,
        help=(
            "Center of the J1 sweep, in degrees. The waypoint table is stored "
            "as J1 offsets; absolute J1 commands = j1_center + offset. "
            "Use +90 if your free workspace is one quarter-turn from home, "
            "-90 for the opposite quarter-turn, or 180 for directly behind home."
        ),
    )
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=0,
        help=(
            "If > 0, stop after this many waypoints. Use 1 for a quick safety "
            "probe: confirm the arm rotates into the free workspace, NOT into "
            "your monitors, before committing to the full ~3-min sweep."
        ),
    )
    parser.add_argument(
        "--j1-center-2",
        type=float,
        default=None,
        help=(
            "Optional SECOND J1 sweep center. If set, the full waypoint table "
            "is run TWICE: once at --j1-center, then again at --j1-center-2. "
            "Doubles wall time but covers two hemispheres for a much better "
            "constrained Kabsch fit. Use only after confirming both sides are "
            "obstacle-free via separate --max-waypoints 1 probes."
        ),
    )
    # Arm reset
    parser.add_argument("--home-speed", type=int, default=40)
    parser.add_argument("--home-timeout-s", type=float, default=12.0)
    parser.add_argument(
        "--release-servos",
        action="store_true",
        help="Release all servos at end (arm will drop). Default: leave engaged.",
    )
    # Diagnostics
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not move arm; just print board detections for ~10s.",
    )
    parser.add_argument(
        "--board-verify-s",
        type=float,
        default=5.0,
        help="Seconds to verify board detection before moving the arm.",
    )
    args = parser.parse_args()

    board_cfg = CharucoBoardConfig(
        squares_x=int(args.squares_x),
        squares_y=int(args.squares_y),
        square_length_m=float(args.square_mm) * 1e-3,
        marker_length_m=float(args.marker_mm) * 1e-3,
        dictionary_name=str(args.dict),
        legacy_pattern=bool(args.legacy_pattern),
    )
    tip_settings = TipDetectorSettings()
    auto_cfg = AutoCalibratorSettings(
        tip_offset_z_m=float(args.tip_offset_mm) * 1e-3,
        frames_per_waypoint=int(args.frames_per_waypoint),
        settle_s=float(args.settle_s),
        move_speed=int(args.move_speed),
        tip_settings=tip_settings,
        j1_center_deg=float(args.j1_center),
    )
    if int(args.max_waypoints) > 0:
        auto_cfg.waypoint_offsets_deg = (
            auto_cfg.waypoint_offsets_deg[: int(args.max_waypoints)]
        )

    j1_min, j1_max = auto_cfg.j1_min_max_deg()
    print(
        f"[auto-cal] J1 range: {j1_min:+.1f} to {j1_max:+.1f} deg "
        f"(center = {auto_cfg.j1_center_deg:+.1f}, {len(auto_cfg.waypoint_offsets_deg)} waypoints)"
    )
    # MyCobot 280's J1 software limit is roughly ±168°. If any waypoint goes
    # past that the arm will silently refuse the command for that row.
    if max(abs(j1_min), abs(j1_max)) > 165.0:
        print(
            f"[auto-cal] WARN: |J1| exceeds 165 deg at the extremes. "
            f"MyCobot 280 limit is ~168 deg; some waypoints may be rejected. "
            f"Consider narrowing the offset table or shifting --j1-center."
        )
    print(
        "[auto-cal] If this points the arm INTO your obstacles, abort (Ctrl-C) "
        "and rerun with --j1-center <new value> (-90, +90, or 180 are typical)."
    )

    # --- Connect to the camera first ---------------------------------------
    source = Record3DSource()
    print(f"[auto-cal] connecting to Record3D device #{args.device_index}...")
    source.connect(device_index=int(args.device_index))

    # --- Verify board detection BEFORE moving the arm ----------------------
    print(
        f"[auto-cal] board: {board_cfg.squares_x}x{board_cfg.squares_y} squares, "
        f"square={board_cfg.square_length_m * 1000:.1f}mm, "
        f"marker={board_cfg.marker_length_m * 1000:.1f}mm, "
        f"dict={board_cfg.dictionary_name}"
    )
    verify_s = float(args.board_verify_s) if not args.dry_run else 10.0
    print(f"[auto-cal] verifying ChArUco detection ({verify_s:.0f}s)...")
    deadline = time.monotonic() + verify_s
    last_print = 0.0
    n_detections = 0
    while time.monotonic() < deadline:
        pkt = source.wait_for_frame(timeout_s=0.25)
        if pkt is None:
            continue
        det = detect_board_pose(pkt.rgb, pkt.intrinsic_mat, board_cfg)
        if det is None:
            continue
        n_detections += 1
        now = time.monotonic()
        if now - last_print > 0.5:
            _print_pose_block("board pose", det.t_camera_board, det)
            last_print = now

    if n_detections == 0:
        print("[auto-cal] FATAL: ChArUco board not detected. Check:")
        print("  - print scale: --square-mm should match the printed square size")
        print("  - dictionary:  --dict should match the print (default DICT_4X4_50)")
        print("  - framing:     board must be fully in iPhone camera view")
        print("  - lighting:    avoid glare/specular reflections on the paper")
        source.disconnect()
        sys.exit(1)
    print(f"[auto-cal] OK: board detected in {n_detections} sample(s)")

    if args.dry_run:
        print("[auto-cal] --dry-run set; not moving the arm. Done.")
        source.disconnect()
        return

    # --- Connect to the arm ------------------------------------------------
    driver = MyCobotDriver(MyCobotDriverSettings(
        port=str(args.port), baudrate=int(args.baudrate)
    ))
    print(f"[auto-cal] connecting to MyCobot on {args.port}...")
    driver.connect()

    try:
        # Reset arm to home BEFORE the waypoint sweep so we start from a known
        # state. Servos may have been released by a previous session, leaving
        # the arm dropped; power_on + home(wait=True) guarantees the first
        # waypoint motion is well-defined.
        print(f"[auto-cal] power_on (currently is_power_on={driver.is_power_on()})")
        driver.power_on()
        print(f"[auto-cal]   is_power_on now = {driver.is_power_on()}")
        try:
            angles_before = driver.get_angles_deg(retries=3)
            print(
                "[auto-cal] angles BEFORE home: "
                f"[{', '.join(f'{a:6.1f}' for a in angles_before)}]"
            )
        except Exception as exc:
            print(f"[auto-cal] could not read pre-home angles: {exc}")

        driver.send_angles_deg([0.0] * 6, speed=int(args.home_speed))
        # Poll-to-converge so we don't proceed mid-motion.
        t_dead = time.monotonic() + float(args.home_timeout_s)
        while time.monotonic() < t_dead:
            try:
                cur = driver.get_angles_deg(retries=2)
            except Exception:
                time.sleep(0.2)
                continue
            if max(abs(a) for a in cur) <= 3.0:
                break
            time.sleep(0.15)
        try:
            angles_after = driver.get_angles_deg(retries=3)
            print(
                "[auto-cal] angles AFTER home:  "
                f"[{', '.join(f'{a:6.1f}' for a in angles_after)}]"
            )
            if max(abs(a) for a in angles_after) > 5.0:
                print(
                    "[auto-cal] WARN: home pose did not fully converge "
                    "(>5 deg from zero). Bump --home-timeout-s or --home-speed."
                )
        except Exception as exc:
            print(f"[auto-cal] could not read post-home angles: {exc}")

        # --- Capture reference depth (arm at home) ----------------------
        # Used by the tip detector to filter out static objects from the
        # depth-peak search. Without this, monitors / robot base / tripod
        # parts higher than the arm tip dominate every frame.
        print("[auto-cal] capturing reference depth (arm at home)...")
        reference_depth = capture_reference_depth(source, n_frames=5)
        if reference_depth is None:
            print(
                "[auto-cal] WARN: could not capture reference depth; "
                "tip detection will fall back to raw depth-peak (less robust)."
            )
        else:
            n_valid = int(np.count_nonzero(np.isfinite(reference_depth) & (reference_depth > 0)))
            print(
                f"[auto-cal]   reference depth: shape={reference_depth.shape}, "
                f"{n_valid} valid pixels"
            )

        # --- Sweep --------------------------------------------------------
        n_offsets = len(auto_cfg.waypoint_offsets_deg)
        sweep_centers = [auto_cfg.j1_center_deg]
        if args.j1_center_2 is not None:
            sweep_centers.append(float(args.j1_center_2))
        total_wp = n_offsets * len(sweep_centers)
        print(
            f"[auto-cal] sweeping {total_wp} waypoints "
            f"({n_offsets} offsets x {len(sweep_centers)} hemisphere(s)): "
            f"j1_centers={sweep_centers}..."
        )
        t0 = time.monotonic()
        samples = []
        for sweep_idx, center in enumerate(sweep_centers):
            if len(sweep_centers) > 1:
                print(
                    f"[auto-cal] === hemisphere {sweep_idx+1}/{len(sweep_centers)}: "
                    f"j1_center={center:+.1f} ==="
                )
                # Re-home BEFORE switching hemispheres so the arm doesn't
                # swing through the obstacle direction on the way over.
                if sweep_idx > 0:
                    print("[auto-cal] re-homing between hemispheres...")
                    driver.send_angles_deg([0.0] * 6, speed=int(args.home_speed))
                    t_dead2 = time.monotonic() + float(args.home_timeout_s)
                    while time.monotonic() < t_dead2:
                        try:
                            cur = driver.get_angles_deg(retries=2)
                        except Exception:
                            time.sleep(0.2)
                            continue
                        if max(abs(a) for a in cur) <= 3.0:
                            break
                        time.sleep(0.15)
            auto_cfg.j1_center_deg = float(center)
            samples.extend(collect_samples(
                driver, source, board_cfg, auto_cfg,
                reference_depth=reference_depth,
            ))
        elapsed = time.monotonic() - t0
        print(
            f"[auto-cal] sweep done in {elapsed:.1f}s; "
            f"{len(samples)}/{total_wp} samples valid"
        )

        if int(args.max_waypoints) > 0 and int(args.max_waypoints) < auto_cfg.min_inlier_samples:
            print(
                f"[auto-cal] --max-waypoints {args.max_waypoints} too few for a fit "
                f"(need {auto_cfg.min_inlier_samples}). Probe done; arm did not collide. "
                "Rerun without --max-waypoints (or with a value >= min_inlier_samples) "
                "to perform the actual calibration."
            )
            return
        if len(samples) < auto_cfg.min_inlier_samples:
            print(
                f"[auto-cal] FATAL: only {len(samples)} valid samples "
                f"(need {auto_cfg.min_inlier_samples}). Likely causes: pointer "
                "out of camera view at most waypoints, board not visible at "
                "most waypoints, or arm not moving."
            )
            sys.exit(2)

        # --- Fit ----------------------------------------------------------
        result = fit_robot_world(samples, auto_cfg)
        ks = result.kabsch
        inlier_set = set(result.inlier_indices)
        print(
            f"[auto-cal] Kabsch RMSE = {ks.rmse_m * 1000:.2f} mm  "
            f"({len(result.inlier_indices)}/{len(samples)} inliers)"
        )
        residuals = list(ks.per_point_residual_m)
        # When the refit dropped outliers, per_point_residual_m only has
        # entries for the inliers. Map back to original sample indices in
        # input order so the printed table aligns.
        if len(residuals) == len(samples):
            for i, s in enumerate(samples):
                tag = "  " if i in inlier_set else " *"
                print(
                    f"[auto-cal] {tag} wp{s.waypoint_idx:02d}: "
                    f"residual={residuals[i] * 1000:5.2f} mm  "
                    f"frames={s.n_frames}"
                )
        else:
            j = 0
            for i, s in enumerate(samples):
                if i in inlier_set:
                    r_mm = residuals[j] * 1000.0
                    j += 1
                    print(
                        f"[auto-cal]    wp{s.waypoint_idx:02d}: "
                        f"residual={r_mm:5.2f} mm  frames={s.n_frames}"
                    )
                else:
                    print(
                        f"[auto-cal]  * wp{s.waypoint_idx:02d}: outlier "
                        f"(dropped pre-refit)  frames={s.n_frames}"
                    )

        # --- Confirm + Save ----------------------------------------------
        target_rmse_m = float(args.max_rmse_mm) * 1e-3
        if ks.rmse_m > target_rmse_m:
            ans = input(
                f"[auto-cal] RMSE={ks.rmse_m * 1000:.2f}mm exceeds target "
                f"{target_rmse_m * 1000:.1f}mm. Save anyway? [y/N] "
            ).strip().lower()
            if ans not in ("y", "yes"):
                print("[auto-cal] not saving.")
                return

        try:
            profile = CalibrationProfileIO.load(args.profile)
            print(f"[auto-cal] loaded existing profile: {args.profile}")
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
            print(f"[auto-cal] created new profile (no prior file at {args.profile})")

        profile.set_robot_world_transform(ks.t_robot_world)
        # World frame == board frame, so the table plane is exactly Z=0 in
        # world. This is geometrically exact (no RANSAC).
        # OpenCV's CharucoBoard convention: board +Z points AWAY from the
        # markers (into the table for a board lying flat). "Above the table"
        # is -Z. Save the table normal as -Z so project_above_table lifts up.
        profile.set_table_plane(
            normal_world=np.array([0.0, 0.0, -1.0]),
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
            "robot_world_rmse_m": float(ks.rmse_m),
            "n_samples": float(len(samples)),
            "n_inliers": float(len(result.inlier_indices)),
            "tip_offset_z_m": float(auto_cfg.tip_offset_z_m),
        }
        profile.created_at_utc = datetime.now(timezone.utc).isoformat()
        CalibrationProfileIO.save(profile, args.profile)
        print(f"[auto-cal] saved profile -> {args.profile}")
        print("[auto-cal] T_robot_world (world->robot) =")
        with np.printoptions(precision=4, suppress=True):
            print(ks.t_robot_world)

    finally:
        if args.release_servos:
            print("[auto-cal] releasing all servos (arm will drop).")
            try:
                driver.release_all_servos()
            except Exception:
                pass
        else:
            print(
                "[auto-cal] keeping servos engaged. To release: rerun with "
                "--release-servos or power-cycle the arm."
            )
        try:
            driver.disconnect()
        finally:
            source.disconnect()


if __name__ == "__main__":
    main()
