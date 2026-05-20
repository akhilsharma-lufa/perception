from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry import ransac_plane, scale_intrinsics_for_shape
from ..io.record3d_source import Record3DSource
from .multitag_calibrator import MultiTagCalibrator, TagObservationFrame
from .profiles import CalibrationProfile, CalibrationProfileIO


@dataclass
class AutoCalibrationSettings:
    target_frames: int = 120
    max_collection_seconds: float = 20.0
    min_unique_tags: int = 2
    profile_path: str = "calibration/profiles/default_session.json"
    depth_subsample_per_frame: int = 1500
    plane_distance_threshold_m: float = 0.008
    plane_min_inlier_ratio: float = 0.30
    plane_ransac_iterations: int = 384


@dataclass
class _DepthSample:
    tag_obs: TagObservationFrame
    pts_camera: np.ndarray  # (N, 3) random subset of valid depth points in camera frame


@dataclass
class CalibrationCollectionResult:
    observations: list[TagObservationFrame] = field(default_factory=list)
    depth_samples: list[_DepthSample] = field(default_factory=list)


class AutoCalibrationManager:
    def __init__(
        self,
        calibrator: MultiTagCalibrator,
        settings: Optional[AutoCalibrationSettings] = None,
    ):
        self.calibrator = calibrator
        self.settings = settings or AutoCalibrationSettings()
        self._rng = np.random.default_rng(0)

    def _subsample_depth_camera_points(
        self,
        depth: np.ndarray,
        intrinsic_mat: np.ndarray,
        rgb_shape: tuple[int, int],
        max_points: int,
    ) -> np.ndarray:
        depth_h, depth_w = depth.shape[:2]
        k_depth = scale_intrinsics_for_shape(
            intrinsic_mat, rgb_shape=rgb_shape, target_shape=(depth_h, depth_w)
        )
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
        ys, xs = np.where(valid)
        if ys.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if ys.size > max_points:
            idx = self._rng.choice(ys.size, size=max_points, replace=False)
            ys = ys[idx]
            xs = xs[idx]
        z = depth[ys, xs].astype(np.float64)
        fx, fy = float(k_depth[0, 0]), float(k_depth[1, 1])
        cx, cy = float(k_depth[0, 2]), float(k_depth[1, 2])
        x = ((xs.astype(np.float64) - cx) * z) / fx
        y = ((ys.astype(np.float64) - cy) * z) / fy
        return np.stack([x, y, z], axis=1)

    def collect_observations(self, source: Record3DSource) -> list[TagObservationFrame]:
        result = self.collect_calibration_data(source)
        return result.observations

    def collect_calibration_data(self, source: Record3DSource) -> CalibrationCollectionResult:
        out = CalibrationCollectionResult()
        start_ts = time.monotonic()
        while len(out.observations) < self.settings.target_frames:
            if time.monotonic() - start_ts > self.settings.max_collection_seconds:
                break
            packet = source.wait_for_frame(timeout_s=0.25)
            if packet is None:
                continue
            obs = self.calibrator.detect_tags(
                rgb=packet.rgb,
                intrinsic_mat=packet.intrinsic_mat,
                ts_monotonic=packet.ts_monotonic,
            )
            out.observations.append(obs)
            pts_cam = self._subsample_depth_camera_points(
                depth=packet.depth,
                intrinsic_mat=packet.intrinsic_mat,
                rgb_shape=packet.rgb.shape[:2],
                max_points=int(self.settings.depth_subsample_per_frame),
            )
            out.depth_samples.append(_DepthSample(tag_obs=obs, pts_camera=pts_cam))
        return out

    def build_and_save_profile(
        self,
        observations_or_data: list[TagObservationFrame] | CalibrationCollectionResult,
    ) -> CalibrationProfile:
        if isinstance(observations_or_data, CalibrationCollectionResult):
            observations = observations_or_data.observations
            depth_samples = observations_or_data.depth_samples
        else:
            observations = observations_or_data
            depth_samples = []

        profile = self.calibrator.bootstrap_profile(observations)

        unique_tags = set()
        for frame in observations:
            unique_tags.update(det.tag_id for det in frame.detections)
        if len(unique_tags) < self.settings.min_unique_tags:
            raise RuntimeError(
                f"Calibration requires >= {self.settings.min_unique_tags} unique tags; got {len(unique_tags)}."
            )

        if depth_samples:
            self._fit_table_plane_into_profile(depth_samples, profile)

        CalibrationProfileIO.save(profile, self.settings.profile_path)
        return profile

    def _fit_table_plane_into_profile(
        self,
        depth_samples: list[_DepthSample],
        profile: CalibrationProfile,
    ) -> None:
        world_chunks: list[np.ndarray] = []
        camera_positions: list[np.ndarray] = []
        for sample in depth_samples:
            estimate = self.calibrator.estimate_world_camera(sample.tag_obs, profile)
            if not estimate.has_world_camera or estimate.t_world_camera is None:
                continue
            if sample.pts_camera.shape[0] == 0:
                continue
            t = estimate.t_world_camera
            pts_world = (sample.pts_camera @ t[:3, :3].T) + t[:3, 3]
            world_chunks.append(pts_world)
            camera_positions.append(t[:3, 3].copy())

        if not world_chunks:
            return

        all_pts = np.concatenate(world_chunks, axis=0)
        cam_centroid = np.mean(np.stack(camera_positions, axis=0), axis=0)

        result = ransac_plane(
            points_xyz=all_pts,
            distance_threshold_m=float(self.settings.plane_distance_threshold_m),
            max_iterations=int(self.settings.plane_ransac_iterations),
            min_inlier_ratio=float(self.settings.plane_min_inlier_ratio),
            seed=0,
            orient_reference=cam_centroid,
        )
        if result is None:
            return
        profile.set_table_plane(
            normal_world=result.normal,
            origin_world=result.origin,
            inlier_ratio=result.inlier_ratio,
            mean_abs_residual_m=result.mean_abs_residual_m,
        )
        profile.metrics["table_plane_inlier_ratio"] = float(result.inlier_ratio)
        profile.metrics["table_plane_mean_abs_residual_m"] = float(result.mean_abs_residual_m)
        # Reset the calibrator's smoothing state so the saved estimates are not biased
        # by the work we did here.
        self.calibrator.reset_runtime_state()

    def evaluate_runtime_geometry_drift(
        self,
        observation: TagObservationFrame,
        profile: CalibrationProfile,
    ) -> float:
        by_id = {det.tag_id: det for det in observation.detections}
        visible_mapped = [tag_id for tag_id in by_id if profile.get_world_tag_transform(tag_id) is not None]
        if len(visible_mapped) < 2:
            return 0.0
        drift_samples = []
        for i, tag_i in enumerate(visible_mapped):
            for tag_j in visible_mapped[i + 1 :]:
                t_cam_i = by_id[tag_i].t_camera_tag
                t_cam_j = by_id[tag_j].t_camera_tag
                t_i_j_obs = np.linalg.inv(t_cam_i) @ t_cam_j

                t_world_i = profile.get_world_tag_transform(tag_i)
                t_world_j = profile.get_world_tag_transform(tag_j)
                if t_world_i is None or t_world_j is None:
                    continue
                t_i_j_expected = np.linalg.inv(t_world_i) @ t_world_j
                drift_samples.append(float(np.linalg.norm(t_i_j_obs[:3, 3] - t_i_j_expected[:3, 3])))
        if not drift_samples:
            return 0.0
        return float(np.mean(drift_samples))
