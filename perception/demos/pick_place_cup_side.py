"""Side-approach pick-and-place for a small (shot-glass) cup.

Differs from `pick_place_cup.py` (top-down) in that the gripper approaches
the cup HORIZONTALLY, with the gripper body lying parallel to the floor and
the fingers cradling the cup from two opposite sides. Designed for objects
with diameter close to or below the gripper's open-finger gap (the AG's
44 mm max open), where a top-down approach is intolerant of XY error.

The sequence:
  1. Detect cup with YOLO + ChArUco anchor (same as the top-down demo).
  2. Move to a "side staging" pose: gripper horizontal, pointing toward the
     cup, fingertips `--side-stand-off-mm` away from the cup along the
     approach axis, at the cup's mid-height in world Z.
  3. Translate forward (along tool0 +Z) into the cup until the fingertips
     surround it.
  4. Close the gripper (the AG's adaptive mechanism stalls at first contact).
  5. Lift vertically by `--lift-mm`.
  6. Translate horizontally to the destination XY (gripper still horizontal,
     RPY unchanged — cup stays vertical because the fingers cradle it from
     the side, gravity keeps it upright).
  7. Lower to release height above the table.
  8. Open the gripper.
  9. Retreat backward along -tool0_Z (gripper backs away from where the cup
     now sits).
 10. Return home.

Run:
    python3 -m perception.demos.pick_place_cup_side --port /dev/ttyUSB0

Defaults assume the cup is approached from the world +X side (i.e., the
gripper points along -world_X). Override with `--approach-axis` if your
setup needs a different direction.
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
    home,
    is_reachable,
    move_to_world,
)
from perception.detection import YoloDetectorSettings, YoloObjectDetector
from perception.geometry.transforms import invert_transform
from perception.io import Record3DSource
from perception.localization import RgbdLocalizerSettings, localize_objects_rgbd


# RPYs for the four cardinal horizontal approach axes.
# Each entry: (gripper points along world axis, RPY (Rx, Ry, Rz) in deg).
# Verified mapping: at RPY=(180,0,0), tool0+Z = world -Z (gripper points down).
# By the XYZ-extrinsic convention, the other horizontal axes map as below.
_APPROACH_RPYS: dict[str, tuple[float, float, float]] = {
    # Gripper fingers point along -world_X (gripper sits at +world_X side of cup)
    "minus-x": (0.0, 90.0, 0.0),
    # Fingers point +X (gripper at -X side of cup)
    "plus-x":  (0.0, -90.0, 0.0),
    # Fingers point -Y (gripper at +Y side of cup)
    "minus-y": (-90.0, 0.0, 0.0),
    # Fingers point +Y (gripper at -Y side of cup)
    "plus-y":  (90.0, 0.0, 0.0),
}

# Approach unit vector (world frame) for each axis. The stand-off position
# sits at cup_xyz + stand_off * approach_vector.
_APPROACH_VECTORS: dict[str, np.ndarray] = {
    "minus-x": np.array([1.0, 0.0, 0.0]),
    "plus-x":  np.array([-1.0, 0.0, 0.0]),
    "minus-y": np.array([0.0, 1.0, 0.0]),
    "plus-y":  np.array([0.0, -1.0, 0.0]),
}


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
            x_mm, y_mm, _ = out_obj.position_world_xyz_m
            key = (int(round(x_mm * 1000.0 / 50.0)), int(round(y_mm * 1000.0 / 50.0)))
            samples[key].append((np.asarray(out_obj.position_world_xyz_m, dtype=np.float64),
                                 float(det.confidence)))

    print(f"[pick_side] observed {n_frames} frames; "
          f"anchor ok={n_anchor_ok} fail={n_anchor_fail}; "
          f"buckets={len(samples)}")
    if not samples:
        return None

    best_key, best_samples = max(
        samples.items(),
        key=lambda item: (len(item[1]), float(np.median([c for _, c in item[1]]))),
    )
    positions = np.stack([p for p, _ in best_samples], axis=0)
    confs = np.asarray([c for _, c in best_samples], dtype=np.float64)
    return (np.median(positions, axis=0), float(np.median(confs)), int(len(best_samples)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-approach pick-and-place for a small cup.")
    parser.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--yolo-model", default="yolo26n-seg.pt")
    parser.add_argument("--classes", default="cup")
    parser.add_argument("--target-label", default="cup")
    parser.add_argument(
        "--place-mm", type=float, nargs=3, default=[40.0, 40.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="World-frame destination XYZ in mm (default 40 40 0)."
    )
    parser.add_argument(
        "--approach-axis", default="minus-x",
        choices=list(_APPROACH_RPYS.keys()),
        help="Direction the gripper FINGERS point along when approaching the cup. "
             "'minus-x' means fingers point along -world_X (gripper body sits to "
             "the +X side of the cup, approaches by moving in -X)."
    )
    parser.add_argument(
        "--side-stand-off-mm", type=float, default=80.0,
        help="Distance from the cup center to the staging pose, along the approach axis (mm)."
    )
    parser.add_argument(
        "--grasp-height-mm", type=float, default=25.0,
        help="World Z height above table at which the gripper engages the cup (mm). "
             "For a 50 mm cup, 25 mm = mid-cup."
    )
    parser.add_argument(
        "--lift-mm", type=float, default=40.0,
        help="How far to lift the cup after grip (mm)."
    )
    parser.add_argument(
        "--release-mm", type=float, default=30.0,
        help="World Z above table at which to release the cup (mm)."
    )
    parser.add_argument(
        "--retreat-mm", type=float, default=80.0,
        help="How far to back the gripper away from the placed cup (mm)."
    )
    parser.add_argument(
        "--grasp-close-value", type=int, default=70,
        help="Gripper close value (0-100). The AG's adaptive mechanism stalls at first contact."
    )
    parser.add_argument(
        "--xy-bias-mm", type=float, nargs=2, default=[11.0, 7.0],
        metavar=("DX", "DY"),
        help="Constant XY offset added to the detected cup world XY (centroid-bias correction)."
    )
    parser.add_argument(
        "--detection-window-s", type=float, default=2.0,
        help="How long to observe before moving."
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Ignore weaker detections."
    )
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--max-reach-mm", type=float, default=320.0)
    parser.add_argument("--squares-x", type=int, default=None)
    parser.add_argument("--squares-y", type=int, default=None)
    parser.add_argument("--square-mm", type=float, default=None)
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--dict", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # --- Profile + context -------------------------------------------------
    profile = CalibrationProfileIO.load(args.profile)
    if profile.robot_world_transform is None or profile.table_plane is None:
        print("[pick_side] ERROR: profile missing robot_world_transform or table_plane.")
        sys.exit(2)

    t_robot_world = profile.get_robot_world_transform()
    n_world, o_world = profile.table_plane.as_arrays()

    settings = MotionSettings(
        max_reach_m=float(args.max_reach_mm) * 1e-3,
        default_speed=int(args.speed),
        # vertical_rpy_deg is unused here (we override per call), but keep default sane.
    )
    ctx = MotionContext(
        t_robot_world=t_robot_world,
        table_normal_world=n_world,
        table_origin_world=o_world,
        settings=settings,
    )

    rpy = _APPROACH_RPYS[args.approach_axis]
    approach_vec = _APPROACH_VECTORS[args.approach_axis]

    board_cfg = _resolve_board_config(profile, args)
    print(f"[pick_side] charuco={board_cfg.squares_x}x{board_cfg.squares_y} "
          f"square={board_cfg.square_length_m*1000:.1f}mm "
          f"marker={board_cfg.marker_length_m*1000:.1f}mm "
          f"dict={board_cfg.dictionary_name} legacy={board_cfg.legacy_pattern}")
    print(f"[pick_side] approach_axis={args.approach_axis}  RPY={rpy}  "
          f"approach_vec_world={approach_vec.tolist()}")

    # --- Detection ---------------------------------------------------------
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

    print("[pick_side] warming YOLO ...")
    t_warm = time.monotonic()
    detector._ensure_model()
    try:
        detector.infer(np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)
    except Exception as exc:
        print(f"[pick_side] warmup inference failed: {exc}")
    print(f"[pick_side] YOLO ready in {time.monotonic() - t_warm:.1f}s")

    source = Record3DSource()
    source.connect(device_index=args.device_index)

    cup_world: np.ndarray | None = None
    try:
        print(f"[pick_side] observing for {args.detection_window_s:.1f}s ...")
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
            print(f"[pick_side] ABORT: no '{args.target_label}' detection.")
            sys.exit(4)
        cup_world, conf, n_samples = result
        print(f"[pick_side] detected cup: world=({cup_world[0]*1000:+.1f}, "
              f"{cup_world[1]*1000:+.1f}, {cup_world[2]*1000:+.1f}) mm  "
              f"conf={conf:.2f}  n_samples={n_samples}")
        bias = np.array([args.xy_bias_mm[0] * 1e-3, args.xy_bias_mm[1] * 1e-3, 0.0])
        if np.any(bias != 0.0):
            cup_world = cup_world + bias
            print(f"[pick_side] after --xy-bias ({args.xy_bias_mm[0]:+.1f}, "
                  f"{args.xy_bias_mm[1]:+.1f}) mm: world=("
                  f"{cup_world[0]*1000:+.1f}, {cup_world[1]*1000:+.1f}, "
                  f"{cup_world[2]*1000:+.1f}) mm")
    finally:
        source.disconnect()

    # --- Compute side-approach waypoints (all in world frame, meters) ------
    # Cup grasp height in world Z. table_normal_world = (0, 0, -1) means
    # "above the table" in world is NEGATIVE Z. So grasp height = -(args.grasp_height_mm/1000).
    grasp_world_z = -float(args.grasp_height_mm) * 1e-3
    release_world_z = -float(args.release_mm) * 1e-3

    cup_xy = np.array([cup_world[0], cup_world[1], grasp_world_z])
    stand_off_m = float(args.side_stand_off_mm) * 1e-3
    staging = cup_xy + approach_vec * stand_off_m              # before approach, gripper offset from cup
    engage = cup_xy                                              # at the cup, fingers around it
    lift_height_world = grasp_world_z - float(args.lift_mm) * 1e-3  # negative Z is "up" in world here
    lifted = np.array([cup_world[0], cup_world[1], lift_height_world])
    place_world = np.asarray(args.place_mm, dtype=np.float64) * 1e-3
    place_xy = np.array([place_world[0], place_world[1], lift_height_world])
    place_release = np.array([place_world[0], place_world[1], release_world_z])
    retreat = place_release + approach_vec * float(args.retreat_mm) * 1e-3

    # --- Reachability ------------------------------------------------------
    for name, pt in [
        ("staging", staging), ("engage", engage), ("lifted", lifted),
        ("place_xy", place_xy), ("place_release", place_release), ("retreat", retreat),
    ]:
        ok, dist = is_reachable(pt, ctx)
        print(f"[pick_side] reach {name}: world=({pt[0]*1000:+.1f}, "
              f"{pt[1]*1000:+.1f}, {pt[2]*1000:+.1f}) mm  tip_norm={dist*1000:.1f}mm  "
              f"{'OK' if ok else 'OUT'}")
        if not ok:
            print(f"[pick_side] ABORT: {name} out of reach. Try a closer --place-mm "
                  f"or smaller --side-stand-off-mm.")
            sys.exit(3)

    print("[pick_side] PLAN:")
    for i, (label, pt) in enumerate([
        ("1 staging        (gripper horizontal)", staging),
        ("2 engage         (fingers around cup)", engage),
        ("[close gripper]", None),
        ("3 lifted         (lift cup)", lifted),
        ("4 place_xy       (translate over destination)", place_xy),
        ("5 place_release  (lower to release height)", place_release),
        ("[open gripper]", None),
        ("6 retreat        (back away from cup)", retreat),
        ("[home]", None),
    ]):
        if pt is None:
            print(f"   {label}")
        else:
            print(f"   {label}: ({pt[0]*1000:+.0f}, {pt[1]*1000:+.0f}, {pt[2]*1000:+.0f}) mm")

    if args.dry_run:
        print("[pick_side] --dry-run; not moving.")
        return

    # --- Execute ----------------------------------------------------------
    driver = MyCobotDriver(MyCobotDriverSettings(port=args.port, baudrate=args.baudrate))
    driver.connect()
    try:
        driver.power_on()
        time.sleep(0.5)
        gripper = Gripper(driver, GripperSettings())

        def _log_pose(tag: str) -> None:
            try:
                p = driver.get_coords_mm_deg(retries=4)
                print(f"[pick_side]   pose AFTER {tag} (mm/deg): "
                      f"({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f}, "
                      f"{p[3]:+.1f}, {p[4]:+.1f}, {p[5]:+.1f})")
            except Exception as exc:
                print(f"[pick_side]   pose AFTER {tag}: read failed: {exc}")

        def _log_gripper(tag: str) -> None:
            try:
                v = gripper.get_value()
                print(f"[pick_side]   gripper AFTER {tag}: value={v} (0=open, 100=closed)")
            except Exception as exc:
                print(f"[pick_side]   gripper AFTER {tag}: read failed: {exc}")

        try:
            print("[pick_side] forcing gripper OPEN ...")
            gripper.open(wait=True)
            time.sleep(0.3)
            _log_gripper("force-open")

            print(f"[pick_side] (1) staging ...")
            move_to_world(driver, staging, ctx, speed=int(args.speed), rpy_deg=rpy)
            _log_pose("staging")

            print(f"[pick_side] (2) engage ...")
            move_to_world(driver, engage, ctx, speed=int(args.speed), rpy_deg=rpy)
            _log_pose("engage")

            print(f"[pick_side] close gripper to value={args.grasp_close_value} ...")
            gripper.close(value=int(args.grasp_close_value), wait=True)
            time.sleep(0.3)
            _log_gripper("close")

            print(f"[pick_side] (3) lift ...")
            move_to_world(driver, lifted, ctx, speed=int(args.speed), rpy_deg=rpy)
            _log_pose("lifted")

            print(f"[pick_side] (4) translate to place XY ...")
            move_to_world(driver, place_xy, ctx, speed=int(args.speed), rpy_deg=rpy)
            _log_pose("place_xy")

            print(f"[pick_side] (5) lower to release ...")
            move_to_world(driver, place_release, ctx, speed=int(args.speed), rpy_deg=rpy)
            _log_pose("place_release")

            print("[pick_side] open gripper ...")
            gripper.open(wait=True)
            time.sleep(0.3)
            _log_gripper("open")

            print(f"[pick_side] (6) retreat ...")
            move_to_world(driver, retreat, ctx, speed=int(args.speed), rpy_deg=rpy)
            _log_pose("retreat")

            print("[pick_side] home ...")
            time.sleep(1.0)
            home(driver, gripper, speed=int(args.speed))
            time.sleep(5.0)
            try:
                angles = driver.get_angles_deg(retries=4)
                print(f"[pick_side]   angles AFTER home: {[round(a, 1) for a in angles]}")
            except Exception as exc:
                print(f"[pick_side]   angles AFTER home: read failed: {exc}")
            print("[pick_side] done.")
        except ReachabilityError as exc:
            print(f"[pick_side] reach error mid-sequence: {exc}")
            sys.exit(3)
    finally:
        driver.disconnect()


if __name__ == "__main__":
    main()
