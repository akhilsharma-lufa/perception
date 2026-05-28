"""Detect a cup with YOLO, pick it up, place it at a world-frame target.

Standalone end-to-end demo:
  1. Opens the iPhone (Record3D) stream.
  2. Observes for `--detection-window-s` seconds, anchoring via ChArUco each
     frame and localizing YOLO detections into world frame.
  3. Picks the best-matching cup (label = `--target-label`, highest median
     confidence across the window).
  4. Runs the existing top-down grasp FSM: pre_grasp -> descend_and_grasp ->
     lift -> place (above target) -> lift -> home.

Run from the repo root, e.g.:
    python3 -m perception.demos.pick_place_cup --port /dev/ttyUSB0

    # Dry-run: detection only, no arm motion.
    python3 -m perception.demos.pick_place_cup --port /dev/ttyUSB0 --dry-run

    # Place at board center instead of origin (useful if origin is out of reach):
    python3 -m perception.demos.pick_place_cup --port /dev/ttyUSB0 --place-mm 100 70 0

Assumptions:
  - `MotionSettings.tip_offset_z_m` is set for the currently-installed tool
    (re-measure with goto_world after a tool change; see CALIBRATION.md).
  - The cup is < 35 mm OD so the AG gripper closes onto it at value=40.
  - The cup is stationary between detection and grasp.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from perception.calibration import CalibrationProfileIO
from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    detect_board_pose,
)
from perception.control import (
    CollisionError,
    Gripper,
    GripperSettings,
    MotionContext,
    MotionSettings,
    MyCobotDriver,
    MyCobotDriverSettings,
    ReachabilityError,
    approach_and_grasp,
    descend_and_grasp,
    is_reachable,
    lift,
    orientation_rpy_for_approach,
    place,
    pre_grasp,
    probe_pose,
    retreat_along_approach,
    safe_home,
)
from perception.control.grasp_planner import (
    GraspInfeasible,
    GraspPlannerSettings,
    GripperGeom,
    plan_grasp,
)
from perception.detection import YoloDetectorSettings, YoloObjectDetector
from perception.geometry.transforms import invert_transform
from perception.io import Record3DSource
from perception.localization import RgbdLocalizerSettings, localize_objects_rgbd


def _resolve_board_config(profile, args) -> CharucoBoardConfig:
    board_from_profile = profile.charuco_board
    return CharucoBoardConfig(
        squares_x=int(
            args.squares_x if args.squares_x is not None
            else (board_from_profile.squares_x if board_from_profile else 11)
        ),
        squares_y=int(
            args.squares_y if args.squares_y is not None
            else (board_from_profile.squares_y if board_from_profile else 8)
        ),
        square_length_m=(
            float(args.square_mm) * 1e-3 if args.square_mm is not None
            else (board_from_profile.square_length_m if board_from_profile else 0.020)
        ),
        marker_length_m=(
            float(args.marker_mm) * 1e-3 if args.marker_mm is not None
            else (board_from_profile.marker_length_m if board_from_profile else 0.014)
        ),
        dictionary_name=str(
            args.dict if args.dict is not None
            else (board_from_profile.dictionary_name if board_from_profile else "DICT_4X4_50")
        ),
        legacy_pattern=bool(
            board_from_profile.legacy_pattern if board_from_profile else True
        ),
    )


def _approach_yaw_toward_base(cup_world_m: np.ndarray, ctx) -> float:
    """In-plane heading (rad) so the angled approach comes from the robot-base side.

    Uses the same plane basis as grasp_planner. Returns the yaw whose +heading
    points from the cup toward the robot base, so the gripper stands off on the
    base side and tilts in over the cup — keeping the arm in its comfortable arc.
    """
    n = np.asarray(ctx.table_normal_world, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    base_world = np.linalg.inv(np.asarray(ctx.t_robot_world, dtype=np.float64))[:3, 3]
    h = np.asarray(base_world, dtype=np.float64) - np.asarray(cup_world_m, dtype=np.float64).reshape(3)
    h = h - np.dot(h, n) * n
    if np.linalg.norm(h) < 1e-6:
        return 0.0
    return float(np.arctan2(np.dot(h, v), np.dot(h, u)))


def _search_feasible_angled_plan(
    cup_world, radius_m, base_radius_m, obj_height_m,
    n_world, o_world, gripper_geom, ctx, tilt_deg, standoff_m, seed_angles,
):
    """Try approach azimuths around the cup; return the first plan whose pre-grasp
    AND grasp poses are IK-solvable and collision-free (probed without moving), or
    None. The user has given freedom on approach direction, so we auto-pick one the
    280 can actually reach."""
    from perception.control.grasp_planner import GraspPlannerSettings, plan_grasp

    base_yaw = _approach_yaw_toward_base(cup_world, ctx)
    # Spread of azimuths to try, relative to the radial-toward-base heading.
    candidates_deg = [0, 180, 90, -90, 45, -45, 135, -135]
    for dyaw in candidates_deg:
        yaw = base_yaw + np.radians(dyaw)
        plan = plan_grasp(
            axis_center_world_m=cup_world, height_m=obj_height_m,
            radius_m=radius_m, base_radius_m=base_radius_m,
            plane_normal_world=n_world, plane_origin_world=o_world,
            gripper=gripper_geom,
            settings=GraspPlannerSettings(approach_tilt_deg=float(tilt_deg),
                                          approach_yaw_rad=float(yaw),
                                          standoff_m=float(standoff_m)),
        )
        rpy = orientation_rpy_for_approach(plan.approach_dir_world, ctx)
        pg_ik, pg_err, pg_col, pg_clr = probe_pose(
            plan.pregrasp_point_world_m, rpy, ctx, seed_angles_deg=seed_angles)
        g_ik, g_err, g_col, g_clr = probe_pose(
            plan.grasp_point_world_m, rpy, ctx, seed_angles_deg=seed_angles)
        feasible = pg_ik and g_ik and pg_col and g_col
        print(f"[pick_place]   azimuth {dyaw:+4d}°: "
              f"pregrasp ik={'ok' if pg_ik else 'FAIL'}({pg_err:.0f}mm) clr={pg_clr*1000:.0f}mm  "
              f"grasp ik={'ok' if g_ik else 'FAIL'}({g_err:.0f}mm) clr={g_clr*1000:.0f}mm  "
              f"-> {'FEASIBLE' if feasible else 'no'}")
        if feasible:
            return plan, dyaw
    return None, None


@dataclass
class CupObservation:
    """Aggregated perception of the target: de-biased world position plus the
    upright object model (radius/height) the grasp planner consumes."""
    position_world_m: np.ndarray
    confidence: float
    n_samples: int
    radius_m: float | None
    base_radius_m: float | None
    height_m: float | None


def _observe_cup(
    source: Record3DSource,
    detector: YoloObjectDetector,
    board_cfg: CharucoBoardConfig,
    table_plane,
    target_label: str,
    min_confidence: float,
    window_s: float,
    known_dims: tuple[float, float, float] | None = None,
    lens_calibration=None,
) -> CupObservation | None:
    """Run detection for `window_s` seconds; return a `CupObservation` for the best
    matching target, or None if no candidate.

    Best = label-match with most samples; ties broken by higher median confidence.
    Position is the median (de-biased axis center) XYZ across the candidate's
    samples; radius/height are medians of the per-frame object-model fits.
    """
    localizer_cfg = RgbdLocalizerSettings()
    # Each sample: (position_xyz, confidence, radius_m|nan, base_radius_m|nan, height_m|nan)
    samples: dict[tuple[int, int], list[tuple[np.ndarray, float, float, float, float]]] = defaultdict(list)

    deadline = time.monotonic() + float(window_s)
    n_frames = 0
    n_anchor_ok = 0
    n_anchor_fail = 0
    while time.monotonic() < deadline:
        packet = source.wait_for_frame(timeout_s=0.25)
        if packet is None:
            continue
        n_frames += 1

        # Lens-corrected PnP when a lens calibration is on the profile; falls
        # back to Record3D's pinhole K when not. This is opt-in: callers that
        # don't pass `lens_calibration` (e.g. the other team's pipeline) get
        # exactly the historical behaviour.
        if lens_calibration is not None:
            k_for_pnp = lens_calibration.k_for_resolution(
                packet.rgb.shape[1], packet.rgb.shape[0]
            )
            dist_for_pnp = lens_calibration.dist_array()
        else:
            k_for_pnp = packet.intrinsic_mat
            dist_for_pnp = None
        board_det = detect_board_pose(
            packet.rgb, k_for_pnp, board_cfg, dist_coeffs=dist_for_pnp,
        )
        if board_det is None:
            n_anchor_fail += 1
            continue
        n_anchor_ok += 1
        t_world_camera = invert_transform(board_det.t_camera_board)

        detections = detector.infer(
            packet.rgb, frame_id=packet.frame_id, camera_pose=packet.camera_pose
        )
        if not detections:
            continue

        outputs = localize_objects_rgbd(
            packet=packet,
            detections=detections,
            t_world_camera=t_world_camera,
            settings=localizer_cfg,
            table_plane=table_plane,
            known_dims=known_dims,
            lens_calibration=lens_calibration,
        )

        # Pair outputs back to detections via "label_index" in object_id.
        for out_obj in outputs:
            try:
                src_idx = int(out_obj.object_id.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if src_idx < 0 or src_idx >= len(detections):
                continue
            det = detections[src_idx]
            if det.label != target_label:
                continue
            if det.confidence < min_confidence:
                continue
            # Bucket by quantized world XY so we keep samples of the SAME cup
            # together across frames (5 cm grid is generous for a 5 cm cup).
            x_mm, y_mm, _ = out_obj.position_world_xyz_m
            key = (int(round(x_mm * 1000.0 / 50.0)), int(round(y_mm * 1000.0 / 50.0)))
            samples[key].append((
                np.asarray(out_obj.position_world_xyz_m, dtype=np.float64),
                float(det.confidence),
                float(out_obj.radius_m) if out_obj.radius_m is not None else float("nan"),
                float(out_obj.base_radius_m) if out_obj.base_radius_m is not None else float("nan"),
                float(out_obj.height_m) if out_obj.height_m is not None else float("nan"),
            ))

    print(f"[pick_place] observed {n_frames} frames; "
          f"anchor ok={n_anchor_ok} fail={n_anchor_fail}; "
          f"buckets={len(samples)}")
    if not samples:
        return None

    def _bucket_score(item):
        _, sample_list = item
        confs = np.asarray([s[1] for s in sample_list], dtype=np.float64)
        return (len(sample_list), float(np.median(confs)))

    best_key, best_samples = max(samples.items(), key=_bucket_score)
    positions = np.stack([s[0] for s in best_samples], axis=0)
    confs = np.asarray([s[1] for s in best_samples], dtype=np.float64)

    def _median_or_none(col: int) -> float | None:
        vals = np.asarray([s[col] for s in best_samples], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else None

    return CupObservation(
        position_world_m=np.median(positions, axis=0),
        confidence=float(np.median(confs)),
        n_samples=int(len(best_samples)),
        radius_m=_median_or_none(2),
        base_radius_m=_median_or_none(3),
        height_m=_median_or_none(4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-driven pick-and-place demo.")
    parser.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--yolo-model", default="yolo26n-seg.pt")
    parser.add_argument(
        "--classes", default="cup",
        help="Comma-separated YOLO labels to accept (filter). Use '*' to disable filtering."
    )
    parser.add_argument(
        "--target-label", default="cup",
        help="Which label to actually pick (must be one of --classes)."
    )
    parser.add_argument(
        "--place-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="World-frame destination XYZ in mm (default 0 0 0 = board origin)."
    )
    parser.add_argument(
        "--hover-mm", type=float, default=50.0,
        help="Transport altitude above the table, in mm (also used for pre_grasp + final back-off). "
             "With the 125 mm AG tool, going much above 60 mm pushes the flange past the arm's "
             "physical reach; 50 mm clears a 50 mm cup with 25 mm of headroom."
    )
    parser.add_argument(
        "--release-mm", type=float, default=30.0,
        help="Release altitude above the table at the destination, in mm."
    )
    parser.add_argument(
        "--grasp-close-value", type=int, default=40,
        help="Gripper close value (0-100). 40 works for ~25-35mm OD cups."
    )
    parser.add_argument(
        "--detection-window-s", type=float, default=2.0,
        help="How long to observe the scene before picking."
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.15,
        help="Ignore detections below this confidence. Lenient default (0.15) "
             "for the red plastic shooter cup, which YOLO scores low. If it gets "
             "mislabelled (not 'cup'), also pass --classes '*' --target-label <label>."
    )
    parser.add_argument(
        "--use-ik", action="store_true",
        help="Use our URDF-based IK + send_angles instead of the firmware's "
             "send_coords Cartesian solver. More reliable; requires a fitted "
             "JointMap (run: ik_debug.py compare --save). See PICK_PLACE.md runbook."
    )
    parser.add_argument(
        "--ik-orient", choices=("none", "Z", "all"), default="Z",
        help="IK orientation constraint (default Z = approach axis; for top-down grasps)."
    )
    parser.add_argument(
        "--xy-bias-mm", type=float, nargs=2, default=[0.0, 0.0],
        metavar=("DX", "DY"),
        help="Constant XY offset added to the detected cup world XY before "
             "the arm goes there (mm). Compensates for YOLO centroid bias — "
             "the mask sees only the camera-facing side of the cup, so the "
             "3D centroid sits toward the camera from the true cup axis. "
             "If the gripper consistently lands 10 mm short toward the camera, "
             "pass --xy-bias-mm 10 8 (or whatever direction takes the centroid "
             "back to the cup's true center)."
    )
    parser.add_argument(
        "--speed", type=int, default=20,
        help="Arm cartesian speed (1-100)."
    )
    parser.add_argument(
        "--max-reach-mm", type=float, default=320.0,
        help="Tip-reach gate (mm). Default 320 accommodates the AG-equipped flange's wider envelope; "
             "the tip itself stays well within physical reach because the 125 mm tool extends downward."
    )
    parser.add_argument(
        "--rpy", type=float, nargs=3, default=None,
        help="Override vertical RPY (deg). Default 180 0 0."
    )
    parser.add_argument("--squares-x", type=int, default=None)
    parser.add_argument("--squares-y", type=int, default=None)
    parser.add_argument("--square-mm", type=float, default=None)
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--dict", default=None)
    # --- Grasp planning (object model -> approach) ---
    parser.add_argument("--gripper-min-gap-mm", type=float, default=20.0,
                        help="Gripper closed finger gap (mm).")
    parser.add_argument("--gripper-max-gap-mm", type=float, default=45.0,
                        help="Gripper open finger gap (mm); the grasp must fit inside this.")
    parser.add_argument("--grip-clearance-mm", type=float, default=6.0,
                        help="Clearance subtracted from max gap so open fingers clear the object.")
    parser.add_argument("--approach-tilt-deg", type=float, default=45.0,
                        help="Tilt from vertical for the angled approach (0=top-down, 90=horizontal).")
    parser.add_argument("--standoff-mm", type=float, default=60.0,
                        help="Pre-grasp standoff distance back along the approach axis (mm).")
    parser.add_argument("--object-rim-mm", type=float, default=None,
                        help="Override: object's max/rim diameter (mm). When set (with "
                             "--object-height-mm), the grasp uses these SUPPLIED dimensions "
                             "and perception provides position only. Reliable for a known "
                             "object when depth-based sizing is unreliable. General: pass the "
                             "dims of whatever you're picking.")
    parser.add_argument("--object-base-mm", type=float, default=None,
                        help="Override: object's base diameter (mm). Defaults to --object-rim-mm "
                             "(cylinder) if omitted.")
    parser.add_argument("--object-height-mm", type=float, default=None,
                        help="Override: object height above the table (mm). Required to use the "
                             "supplied-dimensions path.")
    parser.add_argument("--force-top-down", action="store_true",
                        help="Skip the planner's angled decision and force a vertical grasp "
                             "(only safe when the object fits the open gripper from above).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect and print the plan; do not move the arm."
    )
    args = parser.parse_args()

    # --- Profile + context -------------------------------------------------
    profile = CalibrationProfileIO.load(args.profile)
    if profile.robot_world_transform is None:
        print("[pick_place] ERROR: profile has no robot_world_transform.")
        sys.exit(2)
    if profile.table_plane is None:
        print("[pick_place] ERROR: profile has no table_plane.")
        sys.exit(2)

    t_robot_world = profile.get_robot_world_transform()
    n_world, o_world = profile.table_plane.as_arrays()

    settings = MotionSettings(
        max_reach_m=float(args.max_reach_mm) * 1e-3,
        default_speed=int(args.speed),
        vertical_rpy_deg=(
            tuple(float(v) for v in args.rpy) if args.rpy is not None
            else MotionSettings().vertical_rpy_deg
        ),
        hover_height_m=float(args.hover_mm) * 1e-3,
        release_clearance_m=float(args.release_mm) * 1e-3,
        grasp_close_value=int(args.grasp_close_value),
        use_ik_solver=bool(args.use_ik),
        ik_orientation_mode=str(args.ik_orient),
    )
    ctx = MotionContext(
        t_robot_world=t_robot_world,
        table_normal_world=n_world,
        table_origin_world=o_world,
        settings=settings,
    )

    board_cfg = _resolve_board_config(profile, args)
    print(f"[pick_place] charuco={board_cfg.squares_x}x{board_cfg.squares_y} "
          f"square={board_cfg.square_length_m*1000:.1f}mm "
          f"marker={board_cfg.marker_length_m*1000:.1f}mm "
          f"dict={board_cfg.dictionary_name} legacy={board_cfg.legacy_pattern}")

    # Supplied object dimensions -> (base_radius_m, rim_radius_m, height_m). When
    # given, perception uses them to size the object AND to recover a de-biased
    # axis center (known-cone fit), so the detected position needs no --xy-bias.
    known_dims = None
    if args.object_rim_mm is not None and args.object_height_mm is not None:
        base_mm = args.object_base_mm if args.object_base_mm is not None else args.object_rim_mm
        known_dims = (float(base_mm) * 0.5e-3, float(args.object_rim_mm) * 0.5e-3,
                      float(args.object_height_mm) * 1e-3)

    # --- Detection-only setup ---------------------------------------------
    classes_raw = str(args.classes).strip()
    class_whitelist = None
    if classes_raw != "*":
        class_whitelist = tuple(c.strip() for c in classes_raw.split(",") if c.strip()) or None

    detector = YoloObjectDetector(
        YoloDetectorSettings(
            model_path=args.yolo_model,
            min_confidence=float(args.min_confidence),
            class_whitelist=class_whitelist,
            inference_every_n_frames=1,
            hold_frames=1,
        )
    )

    print("[pick_place] warming YOLO (first run on Jetson can take 30-60s)...")
    t_warm = time.monotonic()
    detector._ensure_model()
    try:
        detector.infer(np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)
    except Exception as exc:
        print(f"[pick_place] warmup inference failed (will retry live): {exc}")
    print(f"[pick_place] YOLO ready in {time.monotonic() - t_warm:.1f}s")

    source = Record3DSource()
    source.connect(device_index=args.device_index)

    cup_world: np.ndarray | None = None
    try:
        print(f"[pick_place] observing for {args.detection_window_s:.1f}s "
              f"(target_label='{args.target_label}', min_conf={args.min_confidence:.2f})...")
        result = _observe_cup(
            source=source,
            detector=detector,
            board_cfg=board_cfg,
            table_plane=profile.table_plane,
            target_label=str(args.target_label),
            min_confidence=float(args.min_confidence),
            window_s=float(args.detection_window_s),
            known_dims=known_dims,
            lens_calibration=profile.lens_calibration,
        )
        if result is None:
            print(f"[pick_place] ABORT: no '{args.target_label}' detection.")
            sys.exit(4)
        cup_world = result.position_world_m
        conf, n_samples = result.confidence, result.n_samples
        obs = result
        rad_str = "n/a" if obs.radius_m is None else f"{obs.radius_m*2000:.1f}mm Ø"
        h_str = "n/a" if obs.height_m is None else f"{obs.height_m*1000:.1f}mm"
        print(f"[pick_place] detected cup: world=({cup_world[0]*1000:+.1f}, "
              f"{cup_world[1]*1000:+.1f}, {cup_world[2]*1000:+.1f}) mm  "
              f"conf={conf:.2f}  n_samples={n_samples}  model: max {rad_str}, height {h_str}")
        # Apply manual XY bias correction (centroid-bias compensation).
        bias_dx_m = float(args.xy_bias_mm[0]) * 1e-3
        bias_dy_m = float(args.xy_bias_mm[1]) * 1e-3
        if bias_dx_m != 0.0 or bias_dy_m != 0.0:
            cup_world = np.array(
                [cup_world[0] + bias_dx_m, cup_world[1] + bias_dy_m, cup_world[2]],
                dtype=np.float64,
            )
            print(f"[pick_place] after --xy-bias ({args.xy_bias_mm[0]:+.1f}, "
                  f"{args.xy_bias_mm[1]:+.1f}) mm: world=("
                  f"{cup_world[0]*1000:+.1f}, {cup_world[1]*1000:+.1f}, "
                  f"{cup_world[2]*1000:+.1f}) mm")
    finally:
        source.disconnect()

    # --- Reachability gate -------------------------------------------------
    place_world = np.asarray(args.place_mm, dtype=np.float64) * 1e-3

    pick_ok, pick_dist = is_reachable(cup_world, ctx)
    place_ok, place_dist = is_reachable(place_world, ctx)
    print(f"[pick_place] reach: pick={pick_dist*1000:.1f}mm {'OK' if pick_ok else 'OUT'}, "
          f"place={place_dist*1000:.1f}mm {'OK' if place_ok else 'OUT'} "
          f"(max {settings.max_reach_m*1000:.1f}mm)")
    if not pick_ok:
        print("[pick_place] ABORT: cup is out of arm reach.")
        sys.exit(3)
    if not place_ok:
        print("[pick_place] ABORT: place target is out of arm reach. "
              "Try --place-mm 100 70 0 (board center).")
        sys.exit(3)

    # --- Object dimensions: supplied override (reliable) or perceived model --
    if args.object_rim_mm is not None and args.object_height_mm is not None:
        radius_m = float(args.object_rim_mm) * 0.5e-3
        base_mm = args.object_base_mm if args.object_base_mm is not None else args.object_rim_mm
        base_radius_m = float(base_mm) * 0.5e-3
        obj_height_m = float(args.object_height_mm) * 1e-3
        print(f"[pick_place] using SUPPLIED object dims: rim Ø{args.object_rim_mm:.0f} "
              f"base Ø{base_mm:.0f} height {args.object_height_mm:.0f} mm "
              f"(perception -> position only)")
    elif obs.radius_m is not None and obs.height_m is not None:
        radius_m = float(obs.radius_m)
        base_radius_m = obs.base_radius_m
        obj_height_m = float(obs.height_m)
    else:
        radius_m = base_radius_m = obj_height_m = None

    # --- Grasp planning: object model -> grasp + approach ------------------
    grasp_plan = None
    if radius_m is not None and not args.force_top_down:
        n_world, o_world = profile.table_plane.as_arrays()
        gripper_geom = GripperGeom(
            min_gap_m=float(args.gripper_min_gap_mm) * 1e-3,
            max_gap_m=float(args.gripper_max_gap_mm) * 1e-3,
            grip_clearance_m=float(args.grip_clearance_mm) * 1e-3,
        )
        # First check feasibility/mode with a straight-down probe plan.
        try:
            probe_plan = plan_grasp(
                axis_center_world_m=cup_world, height_m=obj_height_m,
                radius_m=radius_m, base_radius_m=base_radius_m,
                plane_normal_world=n_world, plane_origin_world=o_world,
                gripper=gripper_geom,
                settings=GraspPlannerSettings(approach_tilt_deg=float(args.approach_tilt_deg),
                                              standoff_m=float(args.standoff_mm) * 1e-3),
            )
        except GraspInfeasible as exc:
            print(f"[pick_place] ABORT: object not graspable: {exc}")
            sys.exit(5)

        if probe_plan.mode == "top_down":
            grasp_plan = probe_plan  # fits the open gripper from above; no search needed
        else:
            # Angled: the tool-collision guard must be on, and we auto-search the
            # approach azimuth for one the arm can actually reach (probed, no motion).
            ctx.settings.enable_collision_check = True
            # The angled grasp needs the FULL orientation enforced — fingers LEVEL
            # (both at the same height) and the servo bump UP/back. ik_orient Z only
            # pins the approach axis and leaves the roll free, which twisted the
            # servo toward the camera and tipped the cup. Force full-orientation IK.
            if ctx.settings.ik_orientation_mode != "all":
                print("[pick_place] angled grasp: forcing full-orientation IK "
                      "(ik_orient=all) so fingers stay level and the servo points up")
                ctx.settings.ik_orientation_mode = "all"
                ctx.ik_solver = None  # rebuild the solver in 'all' mode
            print(f"[pick_place] searching approach azimuths (tilt {args.approach_tilt_deg:.0f}°)...")
            grasp_plan, chosen = _search_feasible_angled_plan(
                cup_world, radius_m, base_radius_m, obj_height_m, n_world, o_world,
                gripper_geom, ctx, float(args.approach_tilt_deg),
                float(args.standoff_mm) * 1e-3, seed_angles=[0.0, -30.0, -30.0, 0.0, 0.0, -45.0],
            )
            if grasp_plan is None:
                print(f"[pick_place] ABORT: no reachable angled approach at tilt "
                      f"{args.approach_tilt_deg:.0f}°. Try a lower --approach-tilt-deg "
                      f"(e.g. 20) or move the cup closer to the base.")
                sys.exit(5)
            print(f"[pick_place] chose approach azimuth {chosen:+d}° (relative to base radial).")
    else:
        reason = ("--force-top-down" if args.force_top_down
                  else "no object dimensions (supply --object-rim-mm/--object-height-mm "
                       "or rely on perception)")
        print(f"[pick_place] no grasp plan ({reason}); using legacy top-down sequence.")

    print(f"[pick_place] PLAN:")
    if grasp_plan is not None:
        print(f"  grasp mode = {grasp_plan.mode.upper()}  "
              f"grasp height {grasp_plan.grasp_height_m*1000:.1f} mm  "
              f"Ø {grasp_plan.grasp_diameter_m*1000:.1f} mm  tilt {grasp_plan.tilt_deg:.0f}°")
        if grasp_plan.notes:
            print(f"  NOTE: {grasp_plan.notes}")
        if grasp_plan.mode == "angled":
            print(f"  1. approach_and_grasp (standoff -> grasp along approach axis, collision-checked)")
            print(f"  2. retreat along approach axis ({args.hover_mm:.0f} mm)")
        else:
            print(f"  1. pre_grasp + descend_and_grasp (vertical)")
            print(f"  2. lift {args.hover_mm:.0f} mm")
    else:
        print(f"  1. pre_grasp     hover {args.hover_mm:.0f} mm above cup")
        print(f"  2. descend_and_grasp  ({ctx.settings.grasp_offset_from_table_m*1000:.0f} mm above table)")
        print(f"  3. lift               {args.hover_mm:.0f} mm")
    print(f"  ->  place at world ({place_world[0]*1000:+.0f}, "
          f"{place_world[1]*1000:+.0f}, {place_world[2]*1000:+.0f}) mm, "
          f"release at {args.release_mm:.0f} mm, then home")

    if args.dry_run:
        print("[pick_place] --dry-run; not moving.")
        return

    # --- Connect arm + execute --------------------------------------------
    driver = MyCobotDriver(MyCobotDriverSettings(port=args.port, baudrate=args.baudrate))
    driver.connect()
    try:
        driver.power_on()
        time.sleep(0.5)
        gripper = Gripper(driver, GripperSettings())

        def _log_pose(tag: str) -> None:
            try:
                p = driver.get_coords_mm_deg(retries=4)
                print(f"[pick_place]   pose AFTER {tag} (mm/deg): "
                      f"({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f}, "
                      f"{p[3]:+.1f}, {p[4]:+.1f}, {p[5]:+.1f})")
            except Exception as exc:
                print(f"[pick_place]   pose AFTER {tag}: read failed: {exc}")

        def _log_gripper(tag: str) -> None:
            try:
                v = gripper.get_value()
                print(f"[pick_place]   gripper AFTER {tag}: value={v} "
                      f"(0=open, 100=closed)")
            except Exception as exc:
                print(f"[pick_place]   gripper AFTER {tag}: read failed: {exc}")

        try:
            # Force the gripper FULLY open (blocking) before any motion. This
            # eliminates a race between gripper.open(wait=False) in pre_grasp
            # and the immediately-following arm motion, which can leave the
            # fingers half-closed during descent.
            print("[pick_place] forcing gripper OPEN (blocking)...")
            gripper.open(wait=True)
            time.sleep(0.3)
            _log_gripper("force-open")

            # Pre-pose to a shoulder-forward / elbow-down config so the IK solver
            # seeds from a good starting pose (seeding from home [0,0,0,0,0,0] can
            # drop ikpy into a 30-40 mm local minimum for tilted angled poses).
            # Mirrors goto_world / goto_cup.
            prepose = [0.0, -30.0, -30.0, 0.0, 0.0, -45.0]
            try:
                print(f"[pick_place] pre-posing to {prepose} ...")
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
                _log_pose("prepose")
            except Exception as exc:
                print(f"[pick_place] WARN: prepose failed: {exc}")

            if grasp_plan is not None and grasp_plan.mode == "angled":
                print("[pick_place] (1/6) angled approach_and_grasp ...")
                approach_and_grasp(driver, gripper, grasp_plan, ctx,
                                   speed=int(args.speed))
                _log_pose("approach_and_grasp")
                _log_gripper("approach_and_grasp")

                print("[pick_place] (2/6) retreat along approach axis ...")
                retreat_along_approach(driver, grasp_plan, ctx,
                                       dist_m=float(args.hover_mm) * 1e-3,
                                       speed=int(args.speed))
                # Collision guard is only needed for the angled grasp near the table;
                # the vertical transit/place below is clear, so relax it.
                ctx.settings.enable_collision_check = False
                _log_pose("retreat")
                _log_gripper("retreat")
            else:
                print("[pick_place] (1/6) pre_grasp ...")
                pre_grasp(driver, gripper, cup_world, ctx,
                          hover_m=float(args.hover_mm) * 1e-3, speed=int(args.speed))
                _log_pose("pre_grasp")
                _log_gripper("pre_grasp")

                print("[pick_place] (2/6) descend_and_grasp ...")
                descend_and_grasp(driver, gripper, cup_world, ctx, speed=int(args.speed))
                _log_pose("descend_and_grasp")
                _log_gripper("descend_and_grasp")

                print("[pick_place] (3/6) lift ...")
                lift(driver, ctx, lift_m=float(args.hover_mm) * 1e-3, speed=int(args.speed))
                _log_pose("lift")
                _log_gripper("lift")

            print("[pick_place] (4/6) place ...")
            place(driver, gripper, place_world, ctx,
                  release_clearance_m=float(args.release_mm) * 1e-3,
                  speed=int(args.speed))
            _log_pose("place")
            _log_gripper("place")

            print("[pick_place] (5/6) gentle back-off ...")
            lift(driver, ctx, lift_m=float(args.hover_mm) * 1e-3, speed=int(args.speed))
            _log_pose("back-off")
            _log_gripper("back-off")

            print("[pick_place] (6/6) home ...")
            # safe_home verifies arrival by joint angle (the firmware's
            # is_moving() reports 0 before the motion begins), opens the
            # gripper, and re-energises servos first. Returns True/False.
            time.sleep(1.5)  # let any residual back-off motion finish first
            homed = safe_home(driver, gripper, speed=int(args.speed))
            _log_pose("home")
            _log_gripper("home")
            print(f"[pick_place] done. (home {'OK' if homed else 'FAILED'})")
        except ReachabilityError as exc:
            print(f"[pick_place] reach error mid-sequence: {exc}")
            print("[pick_place] attempting safe_home before exit ...")
            safe_home(driver, gripper, speed=int(args.speed))
            sys.exit(3)
        except CollisionError as exc:
            print(f"[pick_place] COLLISION guard tripped mid-sequence: {exc}")
            print("[pick_place] attempting safe_home before exit ...")
            safe_home(driver, gripper, speed=int(args.speed))
            sys.exit(6)
    finally:
        driver.disconnect()


if __name__ == "__main__":
    main()
