import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..io.record3d_source import Record3DSource
from .multitag_calibrator import MultiTagCalibrator, TagObservationFrame
from .profiles import CalibrationProfile, CalibrationProfileIO


@dataclass
class AutoCalibrationSettings:
    target_frames: int = 120
    max_collection_seconds: float = 20.0
    min_unique_tags: int = 3
    profile_path: str = "calibration/profiles/default_session.json"


class AutoCalibrationManager:
    def __init__(
        self,
        calibrator: MultiTagCalibrator,
        settings: Optional[AutoCalibrationSettings] = None,
    ):
        self.calibrator = calibrator
        self.settings = settings or AutoCalibrationSettings()

    def collect_observations(self, source: Record3DSource) -> list[TagObservationFrame]:
        observations: list[TagObservationFrame] = []
        start_ts = time.monotonic()
        while len(observations) < self.settings.target_frames:
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
            observations.append(obs)
        return observations

    def build_and_save_profile(self, observations: list[TagObservationFrame]) -> CalibrationProfile:
        profile = self.calibrator.bootstrap_profile(observations)
        unique_tags = set()
        for frame in observations:
            unique_tags.update(det.tag_id for det in frame.detections)
        if len(unique_tags) < self.settings.min_unique_tags:
            raise RuntimeError(
                f"Calibration requires >= {self.settings.min_unique_tags} unique tags; got {len(unique_tags)}."
            )
        CalibrationProfileIO.save(profile, self.settings.profile_path)
        return profile

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
