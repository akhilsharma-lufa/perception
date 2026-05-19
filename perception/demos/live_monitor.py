import argparse
import csv
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from perception.calibration import (
    AutoCalibrationManager,
    AutoCalibrationSettings,
    CalibrationProfileIO,
    MultiTagCalibrator,
    MultiTagCalibratorSettings,
)
from perception.io import Record3DSource
from perception.geometry import invert_transform
from perception.detection.orientation import RotationCode, detect_orientation


def _rotate_display(image: np.ndarray, rot_code: RotationCode) -> np.ndarray:
    """Rotate for display so user sees upright content."""
    if rot_code == RotationCode.CW90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rot_code == RotationCode.CCW90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot_code == RotationCode.ROT180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


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


def _confidence_to_vis(confidence: np.ndarray) -> np.ndarray:
    c = np.asarray(confidence, dtype=np.float32)
    if c.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    c_u8 = np.clip((c / 2.0) * 255.0, 0, 255).astype(np.uint8)
    return c_u8


def _build_hud_lines(
    anchor_mode: str,
    quality: float,
    residual_m: float,
    residual_deg: float,
    drift_m: float,
    visible_tags: list[int],
    event: str,
    orientation_mode: str,
    pose_motion: Tuple[float, float],
    show_world_axes: bool,
    show_world_grid: bool,
):
    orientation_text = orientation_mode
    return [
        f"anchor_mode: {anchor_mode}",
        f"anchor_quality: {quality:.2f}",
        f"residual: {residual_m*100.0:.1f}cm, {residual_deg:.2f}deg",
        f"geometry_drift: {drift_m*100.0:.1f}cm",
        f"visible_tags: {visible_tags}",
        f"camera_motion: {pose_motion[0]*100.0:.1f}cm, {pose_motion[1]:.2f}deg",
        f"orientation: {orientation_text}",
        f"world_axes: {'on' if show_world_axes else 'off'}",
        f"world_grid: {'on' if show_world_grid else 'off'}",
        f"event: {event if event else '-'}",
        "keys: q=quit, v=view, a=axes, g=grid, r=recal",
    ]


def _rotation_angle_deg(r_prev: np.ndarray, r_new: np.ndarray) -> float:
    r_delta = r_prev.T @ r_new
    trace = float(np.trace(r_delta))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(np.degrees(np.arccos(cos_theta)))


def _project_world_point_to_rgb(
    p_world: np.ndarray, t_world_camera: np.ndarray, k_rgb: np.ndarray
) -> tuple[int, int] | None:
    t_camera_world = invert_transform(t_world_camera)
    p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    p_cam = t_camera_world @ p_h
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    fx, fy = float(k_rgb[0, 0]), float(k_rgb[1, 1])
    cx, cy = float(k_rgb[0, 2]), float(k_rgb[1, 2])
    u = (fx * float(p_cam[0]) / z) + cx
    v = (fy * float(p_cam[1]) / z) + cy
    return int(round(u)), int(round(v))


def _draw_world_axes_on_rgb(
    rgb: np.ndarray,
    t_world_camera: np.ndarray | None,
    k_rgb: np.ndarray,
    axis_len_m: float = 0.08,
):
    if t_world_camera is None:
        return
    o = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    x = np.array([axis_len_m, 0.0, 0.0], dtype=np.float64)
    y = np.array([0.0, axis_len_m, 0.0], dtype=np.float64)

    o_px = _project_world_point_to_rgb(o, t_world_camera, k_rgb)
    x_px = _project_world_point_to_rgb(x, t_world_camera, k_rgb)
    y_px = _project_world_point_to_rgb(y, t_world_camera, k_rgb)
    if o_px is None:
        return

    cv2.drawMarker(rgb, o_px, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
    cv2.putText(
        rgb,
        "origin(tag1)",
        (o_px[0] + 8, o_px[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if x_px is not None:
        cv2.line(rgb, o_px, x_px, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            rgb,
            "X",
            (x_px[0] + 6, x_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if y_px is not None:
        cv2.line(rgb, o_px, y_px, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            rgb,
            "Y",
            (y_px[0] + 6, y_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _draw_world_grid_on_rgb(
    rgb: np.ndarray,
    t_world_camera: np.ndarray | None,
    k_rgb: np.ndarray,
    extent_m: float = 0.30,
    step_m: float = 0.05,
):
    if t_world_camera is None:
        return
    if step_m <= 0.0:
        return

    ticks = np.arange(-extent_m, extent_m + 1e-9, step_m, dtype=np.float64)
    line_color = (90, 90, 90)
    axis_x_color = (0, 0, 180)
    axis_y_color = (0, 180, 0)
    text_color = (180, 180, 180)

    # Draw constant-X lines (parallel to Y)
    for x in ticks:
        p0 = np.array([x, -extent_m, 0.0], dtype=np.float64)
        p1 = np.array([x, +extent_m, 0.0], dtype=np.float64)
        p0_px = _project_world_point_to_rgb(p0, t_world_camera, k_rgb)
        p1_px = _project_world_point_to_rgb(p1, t_world_camera, k_rgb)
        if p0_px is None or p1_px is None:
            continue
        color = axis_x_color if abs(float(x)) < 1e-9 else line_color
        thickness = 2 if abs(float(x)) < 1e-9 else 1
        cv2.line(rgb, p0_px, p1_px, color, thickness, cv2.LINE_AA)

    # Draw constant-Y lines (parallel to X)
    for y in ticks:
        p0 = np.array([-extent_m, y, 0.0], dtype=np.float64)
        p1 = np.array([+extent_m, y, 0.0], dtype=np.float64)
        p0_px = _project_world_point_to_rgb(p0, t_world_camera, k_rgb)
        p1_px = _project_world_point_to_rgb(p1, t_world_camera, k_rgb)
        if p0_px is None or p1_px is None:
            continue
        color = axis_y_color if abs(float(y)) < 1e-9 else line_color
        thickness = 2 if abs(float(y)) < 1e-9 else 1
        cv2.line(rgb, p0_px, p1_px, color, thickness, cv2.LINE_AA)

    # Label meter coordinates at selected intersections.
    label_every = max(1, int(round(0.10 / step_m)))  # roughly every 10cm
    for ix, x in enumerate(ticks):
        for iy, y in enumerate(ticks):
            if ix % label_every != 0 or iy % label_every != 0:
                continue
            p = np.array([x, y, 0.0], dtype=np.float64)
            p_px = _project_world_point_to_rgb(p, t_world_camera, k_rgb)
            if p_px is None:
                continue
            cv2.circle(rgb, p_px, 1, (120, 120, 120), -1, cv2.LINE_AA)
            cv2.putText(
                rgb,
                f"{x:+.2f},{y:+.2f}",
                (p_px[0] + 3, p_px[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                text_color,
                1,
                cv2.LINE_AA,
            )


def _add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(
        out,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _render_text_panel(shape: tuple[int, int, int], title: str, lines: list[str]) -> np.ndarray:
    panel = np.zeros(shape, dtype=np.uint8)
    cv2.putText(
        panel,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y = 58
    for line in lines:
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22
    return panel


def _compose_tiled_dashboard(
    rgb: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    stats_lines: list[str],
    world_panel: np.ndarray,
) -> np.ndarray:
    rgb_l = _add_panel_label(rgb, "RGB + Tag Overlay")
    depth_l = _add_panel_label(depth, "Depth")
    conf_l = _add_panel_label(conf, "Confidence")
    stats = _render_text_panel(rgb_l.shape, "Anchor Stats", stats_lines)
    top = np.hstack([rgb_l, depth_l])
    bottom = np.hstack([conf_l, stats])
    world_l = _add_panel_label(world_panel, "World Coordinates (tag1 frame)")
    filler = np.zeros_like(world_l)
    extra = np.hstack([world_l, filler])
    return np.vstack([top, bottom, extra])


def _render_world_coordinates_panel(
    shape: tuple[int, int, int],
    profile,
    visible_tag_ids: list[int],
    origin_tag_id: int,
    live_tag_positions_xy: dict[int, tuple[float, float]] | None = None,
    extent_m: float = 0.35,
) -> np.ndarray:
    panel = np.zeros(shape, dtype=np.uint8)
    h, w = shape[:2]
    cx, cy = w // 2, h // 2
    margin = 26
    scale = min((w / 2 - margin), (h / 2 - margin)) / max(extent_m, 1e-6)

    # grid every 10 cm in world XY plane
    step = 0.10
    ticks = np.arange(-extent_m, extent_m + 1e-9, step, dtype=np.float64)
    for x in ticks:
        px = int(round(cx + x * scale))
        cv2.line(panel, (px, margin), (px, h - margin), (45, 45, 45), 1, cv2.LINE_AA)
    for y in ticks:
        py = int(round(cy - y * scale))
        cv2.line(panel, (margin, py), (w - margin, py), (45, 45, 45), 1, cv2.LINE_AA)

    # world axes
    cv2.line(panel, (margin, cy), (w - margin, cy), (0, 0, 160), 2, cv2.LINE_AA)  # +X red
    cv2.line(panel, (cx, margin), (cx, h - margin), (0, 160, 0), 2, cv2.LINE_AA)  # +Y green
    cv2.putText(panel, "+X", (w - 54, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2, cv2.LINE_AA)
    cv2.putText(panel, "+Y", (cx + 8, margin + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)

    visible_set = set(int(t) for t in visible_tag_ids)
    live_tag_positions_xy = live_tag_positions_xy or {}
    for tag_id_str, t_world_tag in profile.world_tag_transforms.items():
        tag_id = int(tag_id_str)
        if tag_id in live_tag_positions_xy:
            x, y = live_tag_positions_xy[tag_id]
            source = "live"
        else:
            t = np.asarray(t_world_tag, dtype=np.float64)
            x, y = float(t[0, 3]), float(t[1, 3])
            source = "map"
        px = int(round(cx + x * scale))
        py = int(round(cy - y * scale))
        in_bounds = margin <= px <= (w - margin) and margin <= py <= (h - margin)
        if not in_bounds:
            continue
        is_origin = tag_id == int(origin_tag_id)
        is_visible = tag_id in visible_set
        color = (0, 255, 255) if is_origin else ((0, 255, 0) if is_visible else (130, 130, 130))
        radius = 6 if is_origin else 5
        cv2.circle(panel, (px, py), radius, color, -1, cv2.LINE_AA)
        label = f"tag{tag_id} ({x:+.3f},{y:+.3f}) [{source}]"
        cv2.putText(
            panel,
            label,
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    legend = "yellow=origin, green=visible, gray=mapped-only, [live]/[map]"
    cv2.putText(panel, legend, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return panel


def _compute_live_tag_positions_xy(
    obs,
    anchor,
    profile,
    origin_tag_id: int,
) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    by_id = {int(d.tag_id): d for d in obs.detections}
    origin_tag_id = int(origin_tag_id)

    # Best case: origin visible now, compute directly in current frame.
    if origin_tag_id in by_id:
        t_camera_origin = by_id[origin_tag_id].t_camera_tag
        t_origin_camera = invert_transform(t_camera_origin)
        out[origin_tag_id] = (0.0, 0.0)
        for tag_id, det in by_id.items():
            t_origin_tag = t_origin_camera @ det.t_camera_tag
            out[int(tag_id)] = (float(t_origin_tag[0, 3]), float(t_origin_tag[1, 3]))
        return out

    # Fallback: use current world anchor estimate if origin not visible.
    if anchor.t_world_camera is not None:
        for det in obs.detections:
            t_world_tag = anchor.t_world_camera @ det.t_camera_tag
            out[int(det.tag_id)] = (float(t_world_tag[0, 3]), float(t_world_tag[1, 3]))
        if origin_tag_id not in out and profile.get_world_tag_transform(origin_tag_id) is not None:
            out[origin_tag_id] = (0.0, 0.0)
    return out


def _ensure_csv_writer(path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    exists = path_obj.exists()
    handle = path_obj.open("a", newline="")
    writer = csv.writer(handle)
    if not exists:
        writer.writerow(
            [
                "ts_monotonic",
                "frame_id",
                "anchor_mode",
                "anchor_quality",
                "residual_translation_m",
                "residual_rotation_deg",
                "geometry_drift_m",
                "visible_tag_count",
                "event",
                "camera_motion_translation_m",
                "camera_motion_rotation_deg",
            ]
        )
        handle.flush()
    return handle, writer


def main():
    parser = argparse.ArgumentParser(description="Live perception monitor with RGB/depth/anchor health.")
    parser.add_argument(
        "--profile",
        default="calibration/profiles/session_multitag.json",
        help="Calibration profile path.",
    )
    parser.add_argument("--device-index", type=int, default=0, help="Record3D device index.")
    parser.add_argument("--tag-size-m", type=float, default=0.04, help="AprilTag black square size in meters.")
    parser.add_argument("--origin-tag-id", type=int, default=1, help="World origin tag id.")
    parser.add_argument(
        "--log-csv",
        default="",
        help="Optional CSV path for live stability metrics (e.g. logs/perception_live.csv).",
    )
    parser.add_argument(
        "--log-every-n-frames",
        type=int,
        default=1,
        help="Write one CSV row every N frames (default: 1).",
    )
    parser.add_argument(
        "--separate-windows",
        action="store_true",
        help="Show three separate windows instead of a single tiled dashboard.",
    )
    args = parser.parse_args()

    profile = CalibrationProfileIO.load(args.profile)
    calibrator = MultiTagCalibrator(
        MultiTagCalibratorSettings(
            family=profile.tag_family,
            tag_size_m=float(args.tag_size_m),
            origin_tag_id=int(args.origin_tag_id),
        )
    )
    manager = AutoCalibrationManager(
        calibrator=calibrator,
        settings=AutoCalibrationSettings(
            target_frames=120,
            max_collection_seconds=20.0,
            min_unique_tags=3,
            profile_path=args.profile,
        ),
    )
    source = Record3DSource()

    prev_world_camera = None
    last_event_log_ts = 0.0
    frame_count = 0
    separate_windows = bool(args.separate_windows)
    show_world_axes = False
    show_world_grid = False
    csv_handle = None
    csv_writer = None
    if args.log_csv:
        csv_handle, csv_writer = _ensure_csv_writer(args.log_csv)
        print(f"[perception] CSV logging enabled: {args.log_csv}")

    source.connect(device_index=args.device_index)
    try:
        while True:
            packet = source.wait_for_frame(timeout_s=0.25)
            if packet is None:
                continue

            obs = calibrator.detect_tags(packet.rgb, packet.intrinsic_mat, packet.ts_monotonic)
            anchor = calibrator.estimate_world_camera(obs, profile)
            drift_m = manager.evaluate_runtime_geometry_drift(obs, profile)
            live_tag_xy = _compute_live_tag_positions_xy(
                obs=obs,
                anchor=anchor,
                profile=profile,
                origin_tag_id=int(args.origin_tag_id),
            )

            rgb = cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR)
            for det in obs.detections:
                color = (0, 255, 0) if det.tag_id == int(args.origin_tag_id) else (255, 200, 0)
                cx, cy = int(det.center_px[0]), int(det.center_px[1])
                cv2.drawMarker(rgb, (cx, cy), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                cv2.putText(
                    rgb,
                    f"tag:{det.tag_id}",
                    (cx + 6, max(18, cy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if show_world_axes:
                _draw_world_axes_on_rgb(
                    rgb=rgb,
                    t_world_camera=anchor.t_world_camera,
                    k_rgb=packet.intrinsic_mat,
                    axis_len_m=0.08,
                )
            if show_world_grid:
                _draw_world_grid_on_rgb(
                    rgb=rgb,
                    t_world_camera=anchor.t_world_camera,
                    k_rgb=packet.intrinsic_mat,
                    extent_m=0.30,
                    step_m=0.05,
                )

            motion_translation_m = 0.0
            motion_rotation_deg = 0.0
            if anchor.t_world_camera is not None and prev_world_camera is not None:
                motion_translation_m = float(
                    np.linalg.norm(anchor.t_world_camera[:3, 3] - prev_world_camera[:3, 3])
                )
                motion_rotation_deg = _rotation_angle_deg(
                    prev_world_camera[:3, :3], anchor.t_world_camera[:3, :3]
                )
            if anchor.t_world_camera is not None:
                prev_world_camera = anchor.t_world_camera

            residual_m = (
                anchor.residual_translation_m
                if np.isfinite(anchor.residual_translation_m)
                else 0.0
            )
            residual_deg = (
                anchor.residual_rotation_deg if np.isfinite(anchor.residual_rotation_deg) else 0.0
            )

            rot_label = {
                RotationCode.NONE: "portrait",
                RotationCode.CW90: "landscape_cw",
                RotationCode.CCW90: "landscape_ccw",
                RotationCode.ROT180: "rot180",
            }.get(rot_code, "unknown")

            hud_lines = _build_hud_lines(
                anchor_mode=anchor.anchor_mode,
                quality=anchor.quality,
                residual_m=residual_m,
                residual_deg=residual_deg,
                drift_m=drift_m,
                visible_tags=anchor.visible_tag_ids,
                event=anchor.event,
                orientation_mode=rot_label,
                pose_motion=(motion_translation_m, motion_rotation_deg),
                show_world_axes=show_world_axes,
                show_world_grid=show_world_grid,
            )

            depth_vis = _depth_to_colormap(packet.depth)
            conf_vis = _confidence_to_vis(packet.confidence)
            conf_vis = cv2.cvtColor(conf_vis, cv2.COLOR_GRAY2BGR)

            if depth_vis.shape[:2] != rgb.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            if conf_vis.shape[:2] != rgb.shape[:2]:
                conf_vis = cv2.resize(conf_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            rot_code = detect_orientation(packet.rgb, camera_pose=packet.camera_pose)
            rgb_display = _rotate_display(rgb, rot_code)
            depth_display = _rotate_display(depth_vis, rot_code)
            conf_display = _rotate_display(conf_vis, rot_code)
            world_panel = _render_world_coordinates_panel(
                shape=rgb_display.shape,
                profile=profile,
                visible_tag_ids=anchor.visible_tag_ids,
                origin_tag_id=int(args.origin_tag_id),
                live_tag_positions_xy=live_tag_xy,
                extent_m=0.35,
            )

            if separate_windows:
                stats_panel = _render_text_panel(rgb_display.shape, "Anchor Stats", hud_lines)
                cv2.imshow("perception_rgb_live", rgb_display)
                cv2.imshow("perception_depth_live", depth_display)
                cv2.imshow("perception_confidence_live", conf_display)
                cv2.imshow("perception_stats_live", stats_panel)
                cv2.imshow("perception_world_coords_live", world_panel)
            else:
                dashboard = _compose_tiled_dashboard(
                    rgb_display, depth_display, conf_display, hud_lines, world_panel
                )
                cv2.imshow("perception_dashboard", dashboard)

            now = time.monotonic()
            if anchor.event and now - last_event_log_ts > 0.8:
                print(
                    f"[perception] event={anchor.event} mode={anchor.anchor_mode} "
                    f"quality={anchor.quality:.2f} drift_cm={drift_m*100.0:.1f}"
                )
                last_event_log_ts = now

            frame_count += 1
            if csv_writer is not None and frame_count % max(1, int(args.log_every_n_frames)) == 0:
                csv_writer.writerow(
                    [
                        f"{packet.ts_monotonic:.6f}",
                        int(packet.frame_id),
                        anchor.anchor_mode,
                        f"{anchor.quality:.6f}",
                        f"{residual_m:.6f}",
                        f"{residual_deg:.6f}",
                        f"{drift_m:.6f}",
                        int(len(anchor.visible_tag_ids)),
                        anchor.event,
                        f"{motion_translation_m:.6f}",
                        f"{motion_rotation_deg:.6f}",
                    ]
                )
                csv_handle.flush()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                separate_windows = not separate_windows
                cv2.destroyAllWindows()
            elif key == ord("a"):
                show_world_axes = not show_world_axes
            elif key == ord("g"):
                show_world_grid = not show_world_grid
            elif key == ord("r"):
                print("[perception] Recalibration requested. Hold scene still...")
                observations = manager.collect_observations(source)
                try:
                    profile = manager.build_and_save_profile(observations)
                    print(
                        f"[perception] Recalibration complete. mapped_tags={len(profile.world_tag_transforms)} "
                        f"metrics={profile.metrics}"
                    )
                except Exception as exc:
                    print(f"[perception] Recalibration failed: {exc}")
    finally:
        source.disconnect()
        if csv_handle is not None:
            csv_handle.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
