from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional

import numpy as np

from perception.output import ObjectPoseOutput


@dataclass
class WorldTrackerSettings:
    max_match_distance_m: float = 0.10
    max_missed_frames: int = 12
    position_alpha: float = 0.30
    height_alpha: float = 0.20  # only used if height_median_window <= 1
    height_median_window: int = 15  # 0 or 1 disables the median, falls back to EMA
    yaw_alpha: float = 0.25
    quality_alpha: float = 0.30
    min_quality_for_state_update: float = 0.20


@dataclass
class _Track:
    persistent_id: int
    label: str
    position_world: np.ndarray  # (3,)
    height_m: Optional[float]
    yaw_rad: Optional[float]
    quality: float
    height_history: Deque[float] = field(default_factory=deque)
    missed: int = 0
    matched_this_frame: bool = False


@dataclass
class TrackedObjects:
    objects: list[ObjectPoseOutput] = field(default_factory=list)


def _shortest_angle_lerp(prev: float, new: float, alpha: float) -> float:
    diff = (new - prev + np.pi) % (2 * np.pi) - np.pi
    return float(prev + alpha * diff)


class WorldTracker:
    """Persistent per-object tracker keyed on world-XY proximity within label.

    Greedy nearest-neighbour assignment (label-segregated) keeps the implementation
    free of scipy. Cups in this pipeline are typically <= 5 per frame, so quadratic
    matching is fine.
    """

    def __init__(self, settings: Optional[WorldTrackerSettings] = None):
        self.settings = settings or WorldTrackerSettings()
        self._tracks: dict[int, _Track] = {}
        self._next_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, raw_outputs: Iterable[ObjectPoseOutput]) -> list[ObjectPoseOutput]:
        settings = self.settings
        max_dist2 = float(settings.max_match_distance_m) ** 2

        for track in self._tracks.values():
            track.matched_this_frame = False

        outputs = list(raw_outputs)
        detection_to_track: dict[int, int] = {}

        # Build candidate pairs per label and greedily assign by closest XY distance.
        by_label: dict[str, list[int]] = {}
        for det_idx, obj in enumerate(outputs):
            by_label.setdefault(obj.label, []).append(det_idx)

        for label, det_indices in by_label.items():
            track_ids = [tid for tid, tr in self._tracks.items() if tr.label == label]
            if not track_ids or not det_indices:
                continue

            pairs: list[tuple[float, int, int]] = []
            for det_idx in det_indices:
                p = np.asarray(outputs[det_idx].position_world_xyz_m, dtype=np.float64)
                for tid in track_ids:
                    q = self._tracks[tid].position_world
                    d2 = float((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                    if d2 <= max_dist2:
                        pairs.append((d2, det_idx, tid))
            pairs.sort(key=lambda x: x[0])

            used_dets: set[int] = set()
            used_tracks: set[int] = set()
            for _d2, det_idx, tid in pairs:
                if det_idx in used_dets or tid in used_tracks:
                    continue
                used_dets.add(det_idx)
                used_tracks.add(tid)
                detection_to_track[det_idx] = tid

        # Update matched tracks; create new tracks for the rest.
        emitted: list[ObjectPoseOutput] = []
        for det_idx, obj in enumerate(outputs):
            position = np.asarray(obj.position_world_xyz_m, dtype=np.float64).reshape(3)
            if det_idx in detection_to_track:
                track = self._tracks[detection_to_track[det_idx]]
                self._update_track(track, obj, position)
            else:
                window = max(1, int(self.settings.height_median_window))
                history: Deque[float] = deque(maxlen=window)
                if obj.height_m is not None:
                    history.append(float(obj.height_m))
                track = _Track(
                    persistent_id=self._next_id,
                    label=obj.label,
                    position_world=position.copy(),
                    height_m=obj.height_m,
                    yaw_rad=obj.gripper_yaw_hint_rad,
                    quality=float(obj.quality),
                    height_history=history,
                )
                self._tracks[self._next_id] = track
                self._next_id += 1
            track.matched_this_frame = True
            emitted.append(self._emit(obj, track))

        # Age unmatched tracks and drop stale.
        to_drop: list[int] = []
        for tid, track in self._tracks.items():
            if track.matched_this_frame:
                track.missed = 0
                continue
            track.missed += 1
            if track.missed > int(self.settings.max_missed_frames):
                to_drop.append(tid)
        for tid in to_drop:
            self._tracks.pop(tid, None)

        return emitted

    def _update_track(self, track: _Track, obj: ObjectPoseOutput, position: np.ndarray) -> None:
        s = self.settings
        # Always smooth quality so it reflects the latest measurement.
        q_now = float(obj.quality)
        track.quality = float((1.0 - s.quality_alpha) * track.quality + s.quality_alpha * q_now)

        if q_now < float(s.min_quality_for_state_update):
            return  # Treat as transient noise; keep the previous state.

        track.position_world = (1.0 - s.position_alpha) * track.position_world + s.position_alpha * position

        if obj.height_m is not None:
            window = max(1, int(s.height_median_window))
            if track.height_history.maxlen != window:
                track.height_history = deque(track.height_history, maxlen=window)
            track.height_history.append(float(obj.height_m))
            if window > 1:
                track.height_m = float(np.median(np.asarray(track.height_history, dtype=np.float64)))
            elif track.height_m is None:
                track.height_m = float(obj.height_m)
            else:
                track.height_m = float(
                    (1.0 - s.height_alpha) * track.height_m + s.height_alpha * float(obj.height_m)
                )

        if obj.gripper_yaw_hint_rad is not None:
            if track.yaw_rad is None:
                track.yaw_rad = float(obj.gripper_yaw_hint_rad)
            else:
                track.yaw_rad = _shortest_angle_lerp(
                    track.yaw_rad, float(obj.gripper_yaw_hint_rad), float(s.yaw_alpha)
                )

    @staticmethod
    def _emit(template: ObjectPoseOutput, track: _Track) -> ObjectPoseOutput:
        return ObjectPoseOutput(
            object_id=f"{track.label}-{track.persistent_id}",
            label=track.label,
            position_world_xyz_m=(
                float(track.position_world[0]),
                float(track.position_world[1]),
                float(track.position_world[2]),
            ),
            orientation_world_quat_xyzw=template.orientation_world_quat_xyzw,
            gripper_yaw_hint_rad=track.yaw_rad,
            quality=float(track.quality),
            covariance_diag=template.covariance_diag,
            height_m=track.height_m,
            source_mode=template.source_mode,
        )
