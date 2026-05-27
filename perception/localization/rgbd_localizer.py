from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from perception.calibration.profiles import TablePlane
from perception.detection import YoloDetection
from perception.geometry import scale_intrinsics_for_shape
from perception.io.frame_packet import FramePacket
from perception.output import ObjectPoseOutput


@dataclass
class RgbdLocalizerSettings:
    confidence_floor: int = 1
    min_depth_pixels: int = 24
    mask_erosion_px: int = 3
    object_top_percentile: float = 95.0
    object_base_percentile: float = 5.0
    centroid_percentile: float = 50.0
    # Depth-consistency mask refinement: reject mask pixels whose camera-Z is
    # too far BEHIND the cup's front surface. Anchored to the depth p5 of the
    # mask (front of the cup) plus this allowance, so the cup's full body is
    # preserved while distant background (laptop/wall behind the cup) is rejected.
    depth_consistency_max_extent_m: float = 0.18
    depth_consistency_front_percentile: float = 5.0
    depth_consistency_min_keep_ratio: float = 0.40
    # Fallback (no table plane) ring sampling for table_z estimate
    table_ring_px: int = 8
    min_table_ring_pixels: int = 40
    # --- Object geometry model (cone/cylinder on the table plane) ---
    # Slice thickness (m) used to gather rim (top) and base (low) footprint points
    # for the circle fits that recover the object's axis center + radii.
    model_rim_band_m: float = 0.015
    model_base_band_m: float = 0.015
    # Minimum footprint points to attempt a circle fit (else fall back to the
    # radial-extent / centroid estimate).
    model_min_circle_points: int = 12
    # Reject an absurd circle fit (e.g. near-collinear arc) whose radius exceeds
    # this, and fall back. Generous upper bound for table-top objects.
    model_max_radius_m: float = 0.15


def _resize_mask_to_depth(mask_rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    dh, dw = depth_shape
    src = mask_rgb.astype(np.uint8)
    if src.shape[:2] == (dh, dw):
        return src > 0
    resized = cv2.resize(src, (dw, dh), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def _erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    px = int(max(0, px))
    if px == 0:
        return mask
    kernel = np.ones((px * 2 + 1, px * 2 + 1), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded > 0


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Percentile interpolation with non-negative weights."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return float("nan")
    if w.size != v.size:
        w = np.ones_like(v)
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = np.clip(w[order], 0.0, None)
    total = float(np.sum(w_sorted))
    if total <= 0.0:
        return float(np.percentile(v, q))
    cw = np.cumsum(w_sorted)
    target = (q / 100.0) * total
    return float(np.interp(target, cw, v_sorted))


def _unproject_valid(depth: np.ndarray, valid: np.ndarray, k_depth: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64), (ys, xs)
    z = depth[ys, xs].astype(np.float64)
    fx, fy = float(k_depth[0, 0]), float(k_depth[1, 1])
    cx, cy = float(k_depth[0, 2]), float(k_depth[1, 2])
    x = ((xs.astype(np.float64) - cx) * z) / fx
    y = ((ys.astype(np.float64) - cy) * z) / fy
    return np.stack([x, y, z], axis=1), (ys, xs)


def _camera_to_world(pts_cam: np.ndarray, t_world_camera: np.ndarray) -> np.ndarray:
    if pts_cam.shape[0] == 0:
        return pts_cam
    r = np.asarray(t_world_camera[:3, :3], dtype=np.float64)
    t = np.asarray(t_world_camera[:3, 3], dtype=np.float64).reshape(3)
    return (pts_cam @ r.T) + t


def _yaw_from_points_xy(points_xy: np.ndarray, weights: Optional[np.ndarray] = None) -> Optional[float]:
    if points_xy.shape[0] < 20:
        return None
    pts = np.asarray(points_xy, dtype=np.float64)
    if weights is not None and weights.size == pts.shape[0]:
        w = np.clip(weights, 0.0, None)
        wsum = float(np.sum(w))
        if wsum <= 0:
            c = np.mean(pts, axis=0)
        else:
            c = (w[:, None] * pts).sum(axis=0) / wsum
        centered = pts - c
        cov = (centered.T @ (centered * w[:, None])) / max(1.0, wsum)
    else:
        c = np.mean(pts, axis=0)
        centered = pts - c
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[-1] < 1e-9:
        return None
    ratio = float(eigvals[-1]) / max(float(eigvals[0]), 1e-9)
    if ratio < 1.4:
        return None
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.arctan2(axis[1], axis[0]))


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal in-plane axes (u, v) for a plane with `normal`."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    # Pick a reference axis least parallel to n, project it out.
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    return u, v


def _fit_circle_2d(xy: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Algebraic (Kåsa) circle fit. Returns (cx, cy, r) or None.

    Recovers the true center from even a partial arc (e.g. the camera-facing half
    of a cup rim), which a centroid cannot. Ill-conditioned (near-collinear) inputs
    return None so the caller can fall back.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    a_mat = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(a_mat, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0.0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def _slice_radius(
    footprint_uv: np.ndarray,
    center_uv: tuple[float, float],
    min_points: int,
    max_radius_m: float,
) -> Optional[float]:
    """Best radius estimate for a footprint slice: circle fit, else robust radial
    extent (high percentile of distance-to-center)."""
    if footprint_uv.shape[0] == 0:
        return None
    fit = _fit_circle_2d(footprint_uv) if footprint_uv.shape[0] >= min_points else None
    if fit is not None and 0.0 < fit[2] <= max_radius_m:
        return fit[2]
    d = np.linalg.norm(footprint_uv - np.asarray(center_uv, dtype=np.float64), axis=1)
    if d.size == 0:
        return None
    return float(min(np.percentile(d, 90.0), max_radius_m))


def _fit_object_model(
    pts_world: np.ndarray,
    weights: np.ndarray,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
    height_m: float,
    cfg: "RgbdLocalizerSettings",
) -> Optional[tuple[np.ndarray, float, float]]:
    """Fit an upright solid to the object's points on the table plane.

    Returns (axis_center_world_xyz, radius_m, base_radius_m) or None if there is
    not enough geometry. `axis_center_world` is the de-biased object axis projected
    onto the table plane; `radius_m` is the max (rim) radius — the collision bound;
    `base_radius_m` is a low slice's radius. Recovering the center from a circle fit
    removes the camera-facing-half bias of a raw centroid.
    """
    u, v = _plane_basis(plane_normal)
    origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    rel = pts_world - origin
    h = rel @ np.asarray(plane_normal, dtype=np.float64).reshape(3)
    uu = rel @ u
    vv = rel @ v
    footprint = np.column_stack([uu, vv])

    # Rim slice = the widest part. For a cup that's the top; for a cone narrowing
    # upward it's the base. Fit a circle there to get the axis center + max radius.
    rim_mask = h >= (height_m - float(cfg.model_rim_band_m))
    rim_fp = footprint[rim_mask]
    rim_fit = _fit_circle_2d(rim_fp) if rim_fp.shape[0] >= int(cfg.model_min_circle_points) else None
    if rim_fit is not None and 0.0 < rim_fit[2] <= float(cfg.model_max_radius_m):
        cu, cv, _r = rim_fit
    else:
        # Fall back: try a full-footprint circle fit, else weighted centroid.
        full_fit = _fit_circle_2d(footprint) if footprint.shape[0] >= int(cfg.model_min_circle_points) else None
        if full_fit is not None and 0.0 < full_fit[2] <= float(cfg.model_max_radius_m):
            cu, cv = full_fit[0], full_fit[1]
        else:
            w = np.clip(weights, 0.0, None)
            wsum = float(np.sum(w))
            if wsum <= 0.0:
                cu, cv = float(np.mean(uu)), float(np.mean(vv))
            else:
                cu = float(np.sum(w * uu) / wsum)
                cv = float(np.sum(w * vv) / wsum)

    center_uv = (cu, cv)
    radius_m = _slice_radius(rim_fp if rim_fp.shape[0] else footprint, center_uv,
                             int(cfg.model_min_circle_points), float(cfg.model_max_radius_m))
    if radius_m is None:
        radius_m = _slice_radius(footprint, center_uv,
                                 int(cfg.model_min_circle_points), float(cfg.model_max_radius_m))

    base_mask = h <= float(cfg.model_base_band_m)
    base_fp = footprint[base_mask]
    base_radius_m = _slice_radius(base_fp, center_uv,
                                  int(cfg.model_min_circle_points), float(cfg.model_max_radius_m))

    if radius_m is None:
        return None
    axis_center_world = origin + cu * u + cv * v
    return axis_center_world, float(radius_m), (float(base_radius_m) if base_radius_m is not None else float(radius_m))


def _ring_table_z_fallback(
    mask_depth: np.ndarray,
    depth: np.ndarray,
    k_depth: np.ndarray,
    t_world_camera: Optional[np.ndarray],
    ring_px: int,
    min_ring_pixels: int,
) -> Optional[float]:
    ring_px = int(max(1, ring_px))
    kernel = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask_depth.astype(np.uint8), kernel, iterations=1) > 0
    ring = dilated & (~mask_depth)
    valid_ring = ring & np.isfinite(depth) & (depth > 0.0)
    if int(np.count_nonzero(valid_ring)) < int(min_ring_pixels):
        return None
    pts_cam, _ = _unproject_valid(depth, valid_ring, k_depth)
    if pts_cam.shape[0] == 0:
        return None
    if t_world_camera is not None:
        pts = _camera_to_world(pts_cam, t_world_camera)
    else:
        pts = pts_cam
    return float(np.median(pts[:, 2]))


def localize_objects_rgbd(
    packet: FramePacket,
    detections: list[YoloDetection],
    t_world_camera: np.ndarray | None,
    settings: Optional[RgbdLocalizerSettings] = None,
    table_plane: Optional[TablePlane] = None,
) -> list[ObjectPoseOutput]:
    cfg = settings or RgbdLocalizerSettings()
    rgb_h, rgb_w = packet.rgb.shape[:2]
    depth_h, depth_w = packet.depth.shape[:2]
    k_depth = scale_intrinsics_for_shape(
        packet.intrinsic_mat, rgb_shape=(rgb_h, rgb_w), target_shape=(depth_h, depth_w)
    )

    plane_normal: Optional[np.ndarray] = None
    plane_origin: Optional[np.ndarray] = None
    if table_plane is not None and t_world_camera is not None:
        plane_normal, plane_origin = table_plane.as_arrays()

    confidence_floor = int(cfg.confidence_floor)
    finite_depth = np.isfinite(packet.depth) & (packet.depth > 0.0)
    has_confidence = (
        packet.confidence is not None and packet.confidence.shape == packet.depth.shape
    )

    out: list[ObjectPoseOutput] = []
    for idx, det in enumerate(detections):
        mask_depth_full = _resize_mask_to_depth(det.mask_rgb, (depth_h, depth_w))
        mask_depth = _erode_mask(mask_depth_full, cfg.mask_erosion_px)
        if not np.any(mask_depth):
            mask_depth = mask_depth_full  # cup mask was too small to erode

        valid = mask_depth & finite_depth
        if has_confidence:
            valid &= packet.confidence >= confidence_floor

        if int(np.count_nonzero(valid)) < int(cfg.min_depth_pixels):
            continue

        pts_cam, (ys, xs) = _unproject_valid(packet.depth, valid, k_depth)
        if pts_cam.shape[0] < int(cfg.min_depth_pixels):
            continue

        # Depth-consistency refinement: drop mask pixels whose camera-Z is far
        # behind the cup front. Kills "the cup mask leaked onto the laptop"
        # cases where YOLO is permissive but only one depth surface is the cup.
        if cfg.depth_consistency_max_extent_m > 0.0:
            zs = pts_cam[:, 2]
            z_front = float(np.percentile(zs, float(cfg.depth_consistency_front_percentile)))
            z_max_allowed = z_front + float(cfg.depth_consistency_max_extent_m)
            keep = zs <= z_max_allowed
            keep_ratio = float(keep.sum()) / float(zs.size)
            if (
                keep.sum() >= int(cfg.min_depth_pixels)
                and keep_ratio >= float(cfg.depth_consistency_min_keep_ratio)
            ):
                pts_cam = pts_cam[keep]
                ys = ys[keep]
                xs = xs[keep]

        if has_confidence:
            w_per_pt = packet.confidence[ys, xs].astype(np.float64)
            w_per_pt = np.clip(w_per_pt, 1.0, None)
        else:
            w_per_pt = np.ones(pts_cam.shape[0], dtype=np.float64)

        in_world = t_world_camera is not None
        pts = _camera_to_world(pts_cam, t_world_camera) if in_world else pts_cam

        center_x = _weighted_percentile(pts[:, 0], w_per_pt, cfg.centroid_percentile)
        center_y = _weighted_percentile(pts[:, 1], w_per_pt, cfg.centroid_percentile)
        center_z = _weighted_percentile(pts[:, 2], w_per_pt, cfg.centroid_percentile)
        position = (float(center_x), float(center_y), float(center_z))

        height_m: Optional[float] = None
        radius_m: Optional[float] = None
        base_radius_m: Optional[float] = None
        if plane_normal is not None and plane_origin is not None:
            # With a known table plane, signed distance to plane IS height above table.
            # The cup base sits on the table at signed_dist ~= 0; mask erosion may bias
            # the cloud's low percentile upward, so trust the plane as the base reference.
            signed = (pts - plane_origin) @ plane_normal
            top = _weighted_percentile(signed, w_per_pt, cfg.object_top_percentile)
            height_m = float(max(0.0, top))
            # Fit an upright solid: de-biased axis center (circle fit beats a
            # camera-half centroid) + max (rim) radius + base radius.
            model = _fit_object_model(
                pts, w_per_pt, plane_normal, plane_origin, height_m, cfg
            )
            if model is not None:
                axis_center_world, radius_m, base_radius_m = model
                # Replace XY with the de-biased axis; keep centroid Z for display
                # (downstream grasp planning re-projects onto the table plane).
                position = (
                    float(axis_center_world[0]),
                    float(axis_center_world[1]),
                    float(center_z),
                )
        else:
            table_z = _ring_table_z_fallback(
                mask_depth=mask_depth,
                depth=packet.depth,
                k_depth=k_depth,
                t_world_camera=t_world_camera if in_world else None,
                ring_px=cfg.table_ring_px,
                min_ring_pixels=cfg.min_table_ring_pixels,
            )
            z_lo = _weighted_percentile(pts[:, 2], w_per_pt, cfg.object_base_percentile)
            z_hi = _weighted_percentile(pts[:, 2], w_per_pt, cfg.object_top_percentile)
            if table_z is not None:
                d_lo = abs(table_z - z_lo)
                d_hi = abs(z_hi - table_z)
                height_m = float(max(0.0, max(d_lo, d_hi)))
            else:
                height_m = float(max(0.0, z_hi - z_lo))

        is_symmetric = det.yaw_hint_rad is None
        yaw_hint = None if is_symmetric else _yaw_from_points_xy(pts[:, :2], w_per_pt)

        depth_support = int(pts_cam.shape[0])
        depth_mask_area = max(1, int(np.count_nonzero(mask_depth)))
        support_ratio = float(np.clip(depth_support / float(depth_mask_area), 0.0, 1.0))
        # Confidence-weighted quality: support_ratio scaled by mean per-point confidence (normalized).
        if has_confidence:
            mean_conf = float(np.mean(w_per_pt))
            conf_score = float(np.clip((mean_conf - 1.0) / 2.0, 0.0, 1.0))
            quality = float(np.clip(0.5 * support_ratio + 0.5 * conf_score, 0.0, 1.0))
        else:
            quality = support_ratio

        out.append(
            ObjectPoseOutput(
                object_id=f"{det.label}_{idx}",
                label=det.label,
                position_world_xyz_m=position,
                orientation_world_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                gripper_yaw_hint_rad=yaw_hint,
                quality=quality,
                covariance_diag=(0.01, 0.01, 0.02),
                height_m=height_m,
                radius_m=radius_m,
                base_radius_m=base_radius_m,
                source_mode="world" if in_world else "camera",
            )
        )
    return out
