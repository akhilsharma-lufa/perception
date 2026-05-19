import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from ..geometry.transforms import average_transforms, invert_transform, make_transform
from .profiles import CalibrationProfile


@dataclass
class MultiTagCalibratorSettings:
    family: str = "tag36h11"
    tag_size_m: float = 0.04
    origin_tag_id: int = 1
    min_origin_observations: int = 40
    min_tag_observations: int = 12
    max_anchor_residual_m: float = 0.03
    max_anchor_residual_deg: float = 4.0
    hold_last_world_camera_s: float = 0.75
    pose_smoothing_alpha: float = 0.20


@dataclass
class TagDetection:
    tag_id: int
    t_camera_tag: np.ndarray
    decision_margin: float
    center_px: tuple[int, int]


@dataclass
class TagObservationFrame:
    ts_monotonic: float
    detections: List[TagDetection] = field(default_factory=list)


@dataclass
class AnchorEstimate:
    has_world_camera: bool
    t_world_camera: Optional[np.ndarray]
    anchor_mode: str
    visible_tag_ids: List[int]
    candidate_count: int
    residual_translation_m: float
    residual_rotation_deg: float
    quality: float
    event: str = ""


class MultiTagCalibrator:
    """
    Boardless calibration + runtime anchoring:
    - world origin fixed to origin_tag_id (usually tag 1)
    - auxiliary tags used for stability and fallback
    """

    def __init__(self, settings: Optional[MultiTagCalibratorSettings] = None):
        self.settings = settings or MultiTagCalibratorSettings()
        self._detector = None
        self._last_world_camera: Optional[np.ndarray] = None
        self._last_world_camera_ts: float = 0.0
        self._smoothed_world_camera: Optional[np.ndarray] = None

    def _smooth_world_camera(self, t_world_camera: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.settings.pose_smoothing_alpha, 0.0, 1.0))
        if self._smoothed_world_camera is None or alpha <= 0.0:
            self._smoothed_world_camera = np.asarray(t_world_camera, dtype=np.float64)
            return self._smoothed_world_camera

        prev = self._smoothed_world_camera
        new = np.asarray(t_world_camera, dtype=np.float64)
        t = (1.0 - alpha) * prev[:3, 3] + alpha * new[:3, 3]
        r_blend = (1.0 - alpha) * prev[:3, :3] + alpha * new[:3, :3]
        u, _, vt = np.linalg.svd(r_blend)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1.0
            r = u @ vt
        self._smoothed_world_camera = make_transform(r, t)
        return self._smoothed_world_camera

    def _ensure_detector(self):
        if self._detector is not None:
            return
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:
            raise RuntimeError(
                "pupil_apriltags is required for calibration. Install via `pip install pupil-apriltags`."
            ) from exc
        self._detector = Detector(
            families=self.settings.family,
            nthreads=1,
            quad_decimate=1.5,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def detect_tags(
        self, rgb: np.ndarray, intrinsic_mat: np.ndarray, ts_monotonic: Optional[float] = None
    ) -> TagObservationFrame:
        self._ensure_detector()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        k = np.asarray(intrinsic_mat, dtype=np.float64)
        detections = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])),
            tag_size=float(self.settings.tag_size_m),
        )
        out: List[TagDetection] = []
        for det in detections:
            t_camera_tag = make_transform(
                np.asarray(det.pose_R, dtype=np.float64).reshape(3, 3),
                np.asarray(det.pose_t, dtype=np.float64).reshape(3),
            )
            center = tuple(np.asarray(det.center, dtype=np.float64).round().astype(np.int32).tolist())
            out.append(
                TagDetection(
                    tag_id=int(det.tag_id),
                    t_camera_tag=t_camera_tag,
                    decision_margin=float(det.decision_margin),
                    center_px=center,
                )
            )
        return TagObservationFrame(
            ts_monotonic=time.monotonic() if ts_monotonic is None else float(ts_monotonic),
            detections=out,
        )

    def bootstrap_profile(self, observations: Iterable[TagObservationFrame]) -> CalibrationProfile:
        samples_by_tag: Dict[int, List[np.ndarray]] = defaultdict(list)
        origin_seen = 0
        profile = CalibrationProfile.new(
            origin_tag_id=self.settings.origin_tag_id,
            tag_family=self.settings.family,
            tag_size_m=self.settings.tag_size_m,
        )
        profile.set_world_tag_transform(self.settings.origin_tag_id, np.eye(4, dtype=np.float64))

        for frame in observations:
            by_id = {det.tag_id: det for det in frame.detections}
            if self.settings.origin_tag_id not in by_id:
                continue
            origin_seen += 1
            t_camera_origin = by_id[self.settings.origin_tag_id].t_camera_tag
            t_origin_camera = invert_transform(t_camera_origin)
            for tag_id, det in by_id.items():
                if tag_id == self.settings.origin_tag_id:
                    continue
                samples_by_tag[tag_id].append(t_origin_camera @ det.t_camera_tag)

        if origin_seen < self.settings.min_origin_observations:
            raise RuntimeError(
                f"Insufficient origin observations: got {origin_seen}, "
                f"need >= {self.settings.min_origin_observations}."
            )

        retained_tags = 0
        for tag_id, samples in samples_by_tag.items():
            if len(samples) < self.settings.min_tag_observations:
                continue
            profile.set_world_tag_transform(tag_id, average_transforms(samples))
            retained_tags += 1

        profile.metrics = {
            "origin_observations": float(origin_seen),
            "retained_aux_tags": float(retained_tags),
            "total_aux_tags_seen": float(len(samples_by_tag)),
            "avg_samples_per_aux_tag": float(
                np.mean([len(v) for v in samples_by_tag.values()]) if samples_by_tag else 0.0
            ),
        }
        return profile

    @staticmethod
    def _rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
        r_delta = r_a.T @ r_b
        trace = float(np.trace(r_delta))
        cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
        return float(np.degrees(np.arccos(cos_theta)))

    def estimate_world_camera(
        self,
        frame: TagObservationFrame,
        profile: CalibrationProfile,
    ) -> AnchorEstimate:
        visible = [det.tag_id for det in frame.detections]
        candidates: List[np.ndarray] = []
        for det in frame.detections:
            t_world_tag = profile.get_world_tag_transform(det.tag_id)
            if t_world_tag is None:
                continue
            t_tag_camera = invert_transform(det.t_camera_tag)
            candidates.append(t_world_tag @ t_tag_camera)

        if not candidates:
            since_last = frame.ts_monotonic - self._last_world_camera_ts
            if (
                self._last_world_camera is not None
                and since_last <= float(self.settings.hold_last_world_camera_s)
            ):
                return AnchorEstimate(
                    has_world_camera=True,
                    t_world_camera=self._last_world_camera,
                    anchor_mode="hold_last",
                    visible_tag_ids=visible,
                    candidate_count=0,
                    residual_translation_m=0.0,
                    residual_rotation_deg=0.0,
                    quality=0.35,
                    event="ANCHOR_HOLD_LAST",
                )
            return AnchorEstimate(
                has_world_camera=False,
                t_world_camera=None,
                anchor_mode="missing_tags",
                visible_tag_ids=visible,
                candidate_count=0,
                residual_translation_m=float("inf"),
                residual_rotation_deg=float("inf"),
                quality=0.0,
                event="ANCHOR_LOST",
            )

        t_world_camera = average_transforms(candidates)
        residual_trans = []
        residual_rot = []
        for cand in candidates:
            residual_trans.append(float(np.linalg.norm(cand[:3, 3] - t_world_camera[:3, 3])))
            residual_rot.append(self._rotation_angle_deg(cand[:3, :3], t_world_camera[:3, :3]))

        trans_rmse = float(np.sqrt(np.mean(np.square(residual_trans)))) if residual_trans else 0.0
        rot_rmse = float(np.sqrt(np.mean(np.square(residual_rot)))) if residual_rot else 0.0

        trans_quality = max(0.0, 1.0 - (trans_rmse / max(self.settings.max_anchor_residual_m, 1e-6)))
        rot_quality = max(0.0, 1.0 - (rot_rmse / max(self.settings.max_anchor_residual_deg, 1e-6)))
        quality = float(min(1.0, 0.5 * (trans_quality + rot_quality)))

        event = ""
        if (
            trans_rmse > self.settings.max_anchor_residual_m
            or rot_rmse > self.settings.max_anchor_residual_deg
        ):
            event = "ANCHOR_UNSTABLE"

        t_world_camera = self._smooth_world_camera(t_world_camera)
        self._last_world_camera = t_world_camera
        self._last_world_camera_ts = frame.ts_monotonic

        anchor_mode = "origin_plus_aux" if self.settings.origin_tag_id in visible else "aux_only"
        return AnchorEstimate(
            has_world_camera=True,
            t_world_camera=t_world_camera,
            anchor_mode=anchor_mode,
            visible_tag_ids=visible,
            candidate_count=len(candidates),
            residual_translation_m=trans_rmse,
            residual_rotation_deg=rot_rmse,
            quality=quality,
            event=event,
        )
