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

import numpy as np

from perception.calibration import CalibrationProfileIO
from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    detect_board_pose,
)
from perception.control import (
    Gripper,
    GripperSettings,
    MotionContext,
    MotionSettings,
    MyCobotDriver,
    MyCobotDriverSettings,
    ReachabilityError,
    descend_and_grasp,
    home,
    is_reachable,
    lift,
    place,
    pre_grasp,
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


def _observe_cup(
    source: Record3DSource,
    detector: YoloObjectDetector,
    board_cfg: CharucoBoardConfig,
    table_plane,
    target_label: str,
    min_confidence: float,
    window_s: float,
) -> tuple[np.ndarray, float, int] | None:
    """Run detection for `window_s` seconds; return (world_xyz_m, median_conf,
    n_samples) for the best matching target, or None if no candidate.

    Best = label-match with most samples; ties broken by higher median
    confidence. Position is the median XYZ across the candidate's samples.
    """
    localizer_cfg = RgbdLocalizerSettings()
    samples: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = defaultdict(list)

    deadline = time.monotonic() + float(window_s)
    n_frames = 0
    n_anchor_ok = 0
    n_anchor_fail = 0
    while time.monotonic() < deadline:
        packet = source.wait_for_frame(timeout_s=0.25)
        if packet is None:
            continue
        n_frames += 1

        board_det = detect_board_pose(packet.rgb, packet.intrinsic_mat, board_cfg)
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
            samples[key].append((np.asarray(out_obj.position_world_xyz_m, dtype=np.float64),
                                 float(det.confidence)))

    print(f"[pick_place] observed {n_frames} frames; "
          f"anchor ok={n_anchor_ok} fail={n_anchor_fail}; "
          f"buckets={len(samples)}")
    if not samples:
        return None

    def _bucket_score(item):
        _, sample_list = item
        confs = np.asarray([c for _, c in sample_list], dtype=np.float64)
        return (len(sample_list), float(np.median(confs)))

    best_key, best_samples = max(samples.items(), key=_bucket_score)
    positions = np.stack([p for p, _ in best_samples], axis=0)
    confs = np.asarray([c for _, c in best_samples], dtype=np.float64)
    return (
        np.median(positions, axis=0),
        float(np.median(confs)),
        int(len(best_samples)),
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
        "--hover-mm", type=float, default=80.0,
        help="Transport altitude above the table, in mm (also used for pre_grasp + final back-off)."
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
        "--min-confidence", type=float, default=0.5,
        help="Ignore detections below this confidence."
    )
    parser.add_argument(
        "--speed", type=int, default=20,
        help="Arm cartesian speed (1-100)."
    )
    parser.add_argument(
        "--max-reach-mm", type=float, default=270.0,
        help="Conservative reach gate."
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
        )
        if result is None:
            print(f"[pick_place] ABORT: no '{args.target_label}' detection.")
            sys.exit(4)
        cup_world, conf, n_samples = result
        print(f"[pick_place] picked cup: world=({cup_world[0]*1000:+.1f}, "
              f"{cup_world[1]*1000:+.1f}, {cup_world[2]*1000:+.1f}) mm  "
              f"conf={conf:.2f}  n_samples={n_samples}")
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

    print(f"[pick_place] PLAN:")
    print(f"  1. pre_grasp     hover {args.hover_mm:.0f} mm above cup")
    print(f"  2. descend_and_grasp  ({ctx.settings.grasp_offset_from_table_m*1000:.0f} mm above table)")
    print(f"  3. lift               {args.hover_mm:.0f} mm")
    print(f"  4. place              at world ({place_world[0]*1000:+.0f}, "
          f"{place_world[1]*1000:+.0f}, {place_world[2]*1000:+.0f}) mm, "
          f"release at {args.release_mm:.0f} mm")
    print(f"  5. lift               {args.hover_mm:.0f} mm (back off)")
    print(f"  6. home")

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

        try:
            print("[pick_place] (1/6) pre_grasp ...")
            pre_grasp(driver, gripper, cup_world, ctx,
                      hover_m=float(args.hover_mm) * 1e-3, speed=int(args.speed))

            print("[pick_place] (2/6) descend_and_grasp ...")
            descend_and_grasp(driver, gripper, cup_world, ctx, speed=int(args.speed))

            print("[pick_place] (3/6) lift ...")
            lift(driver, ctx, lift_m=float(args.hover_mm) * 1e-3, speed=int(args.speed))

            print("[pick_place] (4/6) place ...")
            place(driver, gripper, place_world, ctx,
                  release_clearance_m=float(args.release_mm) * 1e-3,
                  speed=int(args.speed))

            print("[pick_place] (5/6) gentle back-off ...")
            lift(driver, ctx, lift_m=float(args.hover_mm) * 1e-3, speed=int(args.speed))

            print("[pick_place] (6/6) home ...")
            home(driver, gripper, speed=int(args.speed))
            print("[pick_place] done.")
        except ReachabilityError as exc:
            print(f"[pick_place] reach error mid-sequence: {exc}")
            sys.exit(3)
    finally:
        driver.disconnect()


if __name__ == "__main__":
    main()
