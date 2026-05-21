import argparse

import cv2
import numpy as np

from perception.calibration import (
    AutoCalibrationManager,
    AutoCalibrationSettings,
    CalibrationProfileIO,
    MultiTagCalibrator,
    MultiTagCalibratorSettings,
)
from perception.detection import YoloDetection, YoloDetectorSettings, YoloObjectDetector
from perception.detection.orientation import RotationCode, detect_orientation, rotate_to_upright
from perception.localization import (
    RgbdLocalizerSettings,
    WorldTracker,
    WorldTrackerSettings,
    localize_objects_rgbd,
)
from perception.io import Record3DSource


_DRIFT_WARN_THRESHOLD_M = 0.012  # 12 mm mean tag-to-tag drift starts to matter


def _depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    low = float(np.percentile(d[valid], 5))
    high = float(np.percentile(d[valid], 95))
    if high <= low:
        high = low + 1e-3
    d_norm = np.clip((d - low) / (high - low), 0.0, 1.0)
    d_u8 = (d_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)


_OBJECT_COLORS = [
    (30, 200, 255),
    (100, 255, 100),
    (255, 120, 60),
    (200, 100, 255),
    (60, 255, 200),
]


def _draw_text_box(
    image: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (20, 20, 20),
) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.47
    thickness = 1
    pad_x = 6
    pad_y = 5
    line_gap = 6

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_w = max(w for w, _ in sizes) + 2 * pad_x
    line_h = max(h for _, h in sizes)
    box_h = len(lines) * line_h + (len(lines) - 1) * line_gap + 2 * pad_y

    h, w = image.shape[:2]
    x0 = int(np.clip(x, 0, max(0, w - box_w - 1)))
    y0 = int(np.clip(y, 0, max(0, h - box_h - 1)))
    x1 = x0 + box_w
    y1 = y0 + box_h

    cv2.rectangle(image, (x0, y0), (x1, y1), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x1, y1), (235, 235, 235), 1, cv2.LINE_AA)

    baseline_y = y0 + pad_y + line_h
    for line in lines:
        cv2.putText(
            image,
            line,
            (x0 + pad_x, baseline_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_h + line_gap


def _draw_object_overlays(rgb_bgr: np.ndarray, detections, output_by_det_idx):
    h, w = rgb_bgr.shape[:2]

    if not detections:
        cv2.putText(
            rgb_bgr,
            "no objects detected",
            (w // 2 - 120, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 200),
            2,
            cv2.LINE_AA,
        )
        return

    for idx, det in enumerate(detections):
        color = _OBJECT_COLORS[idx % len(_OBJECT_COLORS)]

        mask = det.mask_rgb.astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = np.zeros_like(rgb_bgr)
        overlay[mask > 0] = color
        rgb_bgr[:] = cv2.addWeighted(rgb_bgr, 1.0, overlay, 0.32, 0.0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_bgr, contours, -1, color, 2, cv2.LINE_AA)

        cx, cy = det.anchor_x, det.anchor_y
        cv2.circle(rgb_bgr, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

        obj = output_by_det_idx.get(idx)
        x, y, bw, _ = cv2.boundingRect(mask)
        panel_lines = [f"{det.label} ({det.confidence:.0%})"]
        if obj is not None:
            panel_lines[0] = f"{obj.object_id} ({det.confidence:.0%})"
            px, py, pz = obj.position_world_xyz_m
            panel_lines.append(f"({px:+.3f}, {py:+.3f}, {pz:+.3f}) m")

            if obj.height_m is not None:
                panel_lines.append(f"h={obj.height_m:.3f} m  q={obj.quality:.2f}")

            if obj.gripper_yaw_hint_rad is not None:
                arrow_len = 30
                dx = int(arrow_len * np.cos(obj.gripper_yaw_hint_rad))
                dy = int(arrow_len * np.sin(obj.gripper_yaw_hint_rad))
                cv2.arrowedLine(
                    rgb_bgr,
                    (cx, cy),
                    (cx + dx, cy + dy),
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )

        panel_x = int(x + bw * 0.5) - 90
        panel_y = max(6, y - 80)
        _draw_text_box(
            rgb_bgr,
            lines=panel_lines,
            x=panel_x,
            y=panel_y,
            text_color=(255, 255, 255),
            bg_color=(18, 18, 18),
        )


def _rotate_display(image: np.ndarray, rot_code: RotationCode) -> np.ndarray:
    if rot_code == RotationCode.CW90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rot_code == RotationCode.CCW90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot_code == RotationCode.ROT180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def _rotate_point_for_display(x: int, y: int, rot_code: RotationCode, src_h: int, src_w: int) -> tuple[int, int]:
    if rot_code == RotationCode.CW90:
        return src_h - 1 - y, x
    if rot_code == RotationCode.CCW90:
        return y, src_w - 1 - x
    if rot_code == RotationCode.ROT180:
        return src_w - 1 - x, src_h - 1 - y
    return x, y


def _rotate_detections_for_display(
    detections: list[YoloDetection],
    rot_code: RotationCode,
    src_h: int,
    src_w: int,
) -> list[YoloDetection]:
    rotated: list[YoloDetection] = []
    for det in detections:
        mask_disp = _rotate_display(det.mask_rgb.astype(np.uint8), rot_code) > 0
        ax, ay = _rotate_point_for_display(det.anchor_x, det.anchor_y, rot_code, src_h, src_w)
        rotated.append(
            YoloDetection(
                label=det.label,
                confidence=det.confidence,
                class_id=det.class_id,
                mask_rgb=mask_disp,
                anchor_x=int(ax),
                anchor_y=int(ay),
                yaw_hint_rad=det.yaw_hint_rad,
            )
        )
    return rotated


def _src_idx_from_object_id(object_id: str) -> int:
    # localizer encodes detection index in object_id as "{label}_{src_idx}"
    try:
        return int(object_id.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def main():
    parser = argparse.ArgumentParser(description="YOLO + RGBD world monitor for cups.")
    parser.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--origin-tag-id", type=int, default=1)
    parser.add_argument("--tag-size-m", type=float, default=0.04)
    parser.add_argument("--yolo-model", default="yolo26n-seg.pt")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument(
        "--classes",
        default="cup,bottle,wine glass,bowl,vase",
        help="Comma-separated class names. Use '*' to disable class filtering.",
    )
    parser.add_argument("--confidence-floor", type=int, default=1)
    parser.add_argument("--infer-every-n", type=int, default=1)
    parser.add_argument("--hold-frames", type=int, default=12)
    parser.add_argument("--smooth-alpha", type=float, default=0.30)
    parser.add_argument("--mask-erosion-px", type=int, default=3)
    parser.add_argument("--match-distance-m", type=float, default=0.10)
    args = parser.parse_args()

    class_whitelist = None
    classes_raw = str(args.classes).strip()
    if classes_raw != "*":
        class_whitelist = tuple(
            c.strip() for c in classes_raw.split(",") if c.strip()
        ) or None

    profile = CalibrationProfileIO.load(args.profile)
    calibrator = MultiTagCalibrator(
        MultiTagCalibratorSettings(
            family=profile.tag_family,
            tag_size_m=float(args.tag_size_m),
            origin_tag_id=int(args.origin_tag_id),
        )
    )
    manager = AutoCalibrationManager(
        calibrator=calibrator, settings=AutoCalibrationSettings(profile_path=args.profile)
    )
    detector = YoloObjectDetector(
        YoloDetectorSettings(
            model_path=args.yolo_model,
            min_confidence=float(args.min_confidence),
            class_whitelist=class_whitelist,
            inference_every_n_frames=max(1, int(args.infer_every_n)),
            hold_frames=max(1, int(args.hold_frames)),
        )
    )
    localizer_cfg = RgbdLocalizerSettings(
        confidence_floor=int(args.confidence_floor),
        mask_erosion_px=int(args.mask_erosion_px),
    )
    tracker = WorldTracker(
        WorldTrackerSettings(
            max_match_distance_m=float(args.match_distance_m),
            position_alpha=float(args.smooth_alpha),
            height_alpha=float(min(args.smooth_alpha, 0.25)),
            yaw_alpha=float(args.smooth_alpha),
            quality_alpha=float(args.smooth_alpha),
        )
    )

    plane_status = "absent"
    if profile.table_plane is not None:
        plane_status = (
            f"OK inliers={profile.table_plane.inlier_ratio:.2f} "
            f"resid={profile.table_plane.mean_abs_residual_m * 1000:.1f}mm"
        )
    source = Record3DSource()
    print(
        f"[perception] YOLO model={args.yolo_model} "
        f"min_conf={float(args.min_confidence):.2f} "
        f"infer_every_n={int(args.infer_every_n)} hold_frames={int(args.hold_frames)} "
        f"classes={class_whitelist if class_whitelist is not None else '*'}"
    )
    print(f"[perception] table_plane={plane_status}")
    if profile.table_plane is None:
        print(
            "[perception] WARN: no table plane in profile. Heights will use ring-fallback "
            "(noisier). Re-run auto_calibrate_tags to populate it."
        )

    # Pre-warm YOLO so the first frame in the live loop is not a 30-60s freeze on Jetson.
    # The first inference triggers PyTorch CUDA init + kernel JIT compilation; we do that
    # here on a dummy frame so the GUI starts responsive.
    import time as _time
    print("[perception] loading + warming YOLO (first run on Jetson can take 30-60s)...")
    _t_warm = _time.monotonic()
    detector._ensure_model()  # forces model load
    try:
        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.infer(_dummy, frame_id=0)
    except Exception as _e:
        print(f"[perception] warmup inference failed (will retry live): {_e}")
    print(f"[perception] YOLO ready in {_time.monotonic() - _t_warm:.1f}s")

    _last_rot_code = RotationCode.NONE
    source.connect(device_index=args.device_index)
    _stage_ms: dict[str, list[float]] = {
        k: [] for k in ("fetch", "apriltag", "anchor", "drift", "yolo",
                        "localize", "tracker", "render", "loop")
    }
    _stage_log_every = 30
    _loop_t_prev = _time.perf_counter()
    try:
        while True:
            _loop_t0 = _time.perf_counter()
            _t = _time.perf_counter()
            packet = source.wait_for_frame(timeout_s=0.25)
            _stage_ms["fetch"].append((_time.perf_counter() - _t) * 1000.0)
            if packet is None:
                continue

            _t = _time.perf_counter()
            obs = calibrator.detect_tags(packet.rgb, packet.intrinsic_mat, packet.ts_monotonic)
            _stage_ms["apriltag"].append((_time.perf_counter() - _t) * 1000.0)

            _t = _time.perf_counter()
            anchor = calibrator.estimate_world_camera(obs, profile)
            _stage_ms["anchor"].append((_time.perf_counter() - _t) * 1000.0)

            _t = _time.perf_counter()
            drift_m = manager.evaluate_runtime_geometry_drift(obs, profile)
            _stage_ms["drift"].append((_time.perf_counter() - _t) * 1000.0)

            _t = _time.perf_counter()
            detections = detector.infer(packet.rgb, frame_id=packet.frame_id, camera_pose=packet.camera_pose)
            _stage_ms["yolo"].append((_time.perf_counter() - _t) * 1000.0)

            _t = _time.perf_counter()
            raw_outputs = localize_objects_rgbd(
                packet=packet,
                detections=detections,
                t_world_camera=anchor.t_world_camera,
                settings=localizer_cfg,
                table_plane=profile.table_plane,
            )
            _stage_ms["localize"].append((_time.perf_counter() - _t) * 1000.0)

            _t = _time.perf_counter()
            src_indices = [_src_idx_from_object_id(o.object_id) for o in raw_outputs]
            tracked_outputs = tracker.update(raw_outputs)
            _stage_ms["tracker"].append((_time.perf_counter() - _t) * 1000.0)
            output_by_det_idx = {
                src_indices[i]: tracked_outputs[i]
                for i in range(len(tracked_outputs))
                if 0 <= src_indices[i] < len(detections)
            }

            rgb_bgr = cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR)
            depth_bgr = _depth_to_colormap(packet.depth)
            if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
                depth_bgr = cv2.resize(depth_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            for det in obs.detections:
                tag_color = (0, 255, 0) if det.tag_id == int(args.origin_tag_id) else (255, 200, 0)
                tcx, tcy = int(det.center_px[0]), int(det.center_px[1])
                cv2.drawMarker(rgb_bgr, (tcx, tcy), tag_color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                cv2.putText(
                    rgb_bgr,
                    f"tag:{det.tag_id}",
                    (tcx + 6, max(18, tcy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    tag_color,
                    2,
                    cv2.LINE_AA,
                )

            rot_code = detect_orientation(packet.rgb, camera_pose=packet.camera_pose)
            if rot_code != _last_rot_code:
                rot_names = {RotationCode.NONE: "portrait", RotationCode.CW90: "landscape_cw",
                             RotationCode.CCW90: "landscape_ccw", RotationCode.ROT180: "rot180"}
                print(f"[perception] orientation changed: {rot_names.get(rot_code, rot_code)}  "
                      f"frame={packet.rgb.shape[1]}x{packet.rgb.shape[0]}  "
                      f"pose=({packet.camera_pose.qx:.3f},{packet.camera_pose.qy:.3f},"
                      f"{packet.camera_pose.qz:.3f},{packet.camera_pose.qw:.3f})")
                _last_rot_code = rot_code
            rgb_display = _rotate_display(rgb_bgr, rot_code)
            depth_display = _rotate_display(depth_bgr, rot_code)
            detections_display = _rotate_detections_for_display(
                detections=detections,
                rot_code=rot_code,
                src_h=packet.rgb.shape[0],
                src_w=packet.rgb.shape[1],
            )

            _draw_object_overlays(rgb_display, detections_display, output_by_det_idx)
            drift_status = "-"
            drift_color = (255, 255, 255)
            if drift_m > 0.0:
                drift_status = f"{drift_m * 1000:.1f}mm"
                if drift_m > _DRIFT_WARN_THRESHOLD_M:
                    drift_color = (60, 80, 255)  # red-ish in BGR
                    drift_status += " HIGH"
            status_line = (
                f"anchor={anchor.anchor_mode} q={anchor.quality:.2f} "
                f"event={anchor.event if anchor.event else '-'} "
                f"drift={drift_status} plane={'on' if profile.table_plane else 'off'}"
            )
            cv2.putText(
                rgb_display,
                status_line,
                (12, rgb_display.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                drift_color,
                2,
                cv2.LINE_AA,
            )

            _t = _time.perf_counter()
            cv2.imshow("perception_yolo_rgb", rgb_display)
            cv2.imshow("perception_yolo_depth", depth_display)

            key = cv2.waitKey(1) & 0xFF
            _stage_ms["render"].append((_time.perf_counter() - _t) * 1000.0)
            _stage_ms["loop"].append((_time.perf_counter() - _loop_t0) * 1000.0)

            if len(_stage_ms["loop"]) >= _stage_log_every:
                now = _time.perf_counter()
                fps = _stage_log_every / max(1e-6, now - _loop_t_prev)
                _loop_t_prev = now
                parts = []
                for k in ("fetch", "apriltag", "anchor", "drift", "yolo",
                          "localize", "tracker", "render", "loop"):
                    vals = _stage_ms[k]
                    if not vals:
                        continue
                    arr = np.asarray(vals, dtype=np.float64)
                    parts.append(f"{k}={arr.mean():.0f}/{np.percentile(arr, 95):.0f}")
                    _stage_ms[k].clear()
                print(f"[loop] fps={fps:.1f}  " + "  ".join(parts) + "   (mean/p95 ms)")

            if key in (ord("q"), 27):
                break
    finally:
        source.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
