"""Cup locator: blocking call returns one stable cup pose in robot mm.

Composes the existing perception stack behind a single class so a path-planning
teammate doesn't need to know about ChArUco anchors, YOLO settings, or world
tracking. See `cup_locator/README.md` for the quickstart.
"""
from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    detect_board_pose,
)
from perception.calibration.profiles import (
    CalibrationProfileIO,
    CharucoBoardSpec,
)
from perception.control.motion_primitives import world_to_robot
from perception.detection.yolo_objects import (
    YoloDetectorSettings,
    YoloObjectDetector,
)
from perception.io.record3d_source import Record3DSource
from perception.localization.rgbd_localizer import (
    RgbdLocalizerSettings,
    localize_objects_rgbd,
)
from perception.localization.world_tracker import (
    WorldTracker,
    WorldTrackerSettings,
)
from perception.output import ObjectPoseOutput


@dataclass(frozen=True)
class CupPose:
    """One smoothed cup pose, expressed in two frames.

    `position_robot_mm` is the number the path planner consumes — it matches
    the MyCobot 280's `get_coords_mm_deg()` / `send_coords_mm_deg()` frame, so
    you can pass it straight into a `send_coords` call with the appropriate
    RPY (e.g. (180, 0, 0) for a top-down grasp). `position_world_m` is the
    board-frame pose the same point came from, kept for debugging.
    """
    position_robot_mm: Tuple[float, float, float]
    position_world_m: Tuple[float, float, float]
    height_m: Optional[float]
    quality: float
    yaw_hint_rad: Optional[float]
    label: str
    track_id: str


class CupLocator(AbstractContextManager["CupLocator"]):
    """Self-contained cup detector + locator.

    Loads a calibrated profile on construction; opens the Record3D camera
    lazily on the first ``locate()`` call. The full perception pipeline
    (ChArUco anchor + YOLO segmentation + RGB-D percentile localization +
    world-frame tracker with median height smoothing) runs inside ``locate``
    and returns once the cup has been seen for ``settle_frames`` consecutive
    frames at acceptable quality, or returns None on timeout.

    Example:

        with CupLocator("calibration/profiles/session_multitag.json") as loc:
            pose = loc.locate(timeout_s=2.0)
            if pose is not None:
                x_mm, y_mm, z_mm = pose.position_robot_mm
    """

    def __init__(
        self,
        profile_path: str,
        *,
        target_label: str = "cup",
        min_confidence: float = 0.15,
        settle_frames: int = 8,
        min_quality: float = 0.20,
        anchor_max_reproj_px: float = 1.5,
        model_path: Optional[str] = None,
    ):
        self._profile_path = str(profile_path)
        profile = CalibrationProfileIO.load(self._profile_path)

        t_robot_world = profile.get_robot_world_transform()
        if t_robot_world is None:
            raise ValueError(
                f"profile {self._profile_path!r} has no robot_world_transform; "
                "run `python -m perception.cup_locator.calibrate ...` first."
            )
        if profile.charuco_board is None:
            raise ValueError(
                f"profile {self._profile_path!r} has no charuco_board geometry; "
                "re-run the calibrate CLI so the board spec is saved."
            )

        self._profile = profile
        self._t_robot_world = np.asarray(t_robot_world, dtype=np.float64).reshape(4, 4)
        self._board_cfg = self._board_cfg_from_spec(profile.charuco_board)

        yolo_settings = YoloDetectorSettings(
            min_confidence=float(min_confidence),
            class_whitelist=(str(target_label),),
        )
        if model_path is not None:
            yolo_settings.model_path = str(model_path)
        self._detector = YoloObjectDetector(yolo_settings)
        self._localizer_cfg = RgbdLocalizerSettings()
        self._tracker = WorldTracker(WorldTrackerSettings())

        self._target_label = str(target_label)
        self._settle_frames = int(max(1, settle_frames))
        self._min_quality = float(min_quality)
        self._anchor_max_reproj_px = float(anchor_max_reproj_px)

        self._source: Optional[Record3DSource] = None
        self._owns_source: bool = False
        self._consecutive_hits: dict[str, int] = {}

    @staticmethod
    def _board_cfg_from_spec(spec: CharucoBoardSpec) -> CharucoBoardConfig:
        return CharucoBoardConfig(
            squares_x=int(spec.squares_x),
            squares_y=int(spec.squares_y),
            square_length_m=float(spec.square_length_m),
            marker_length_m=float(spec.marker_length_m),
            dictionary_name=str(spec.dictionary_name),
            legacy_pattern=bool(spec.legacy_pattern),
        )

    def _ensure_source(self, device_index: int = 0) -> Record3DSource:
        if self._source is None:
            src = Record3DSource()
            src.connect(device_index=int(device_index))
            self._source = src
            self._owns_source = True
        return self._source

    def locate(
        self,
        timeout_s: float = 2.0,
        settle_frames: Optional[int] = None,
        *,
        device_index: int = 0,
    ) -> Optional[CupPose]:
        """Block until one stable cup pose is available, or return None on timeout.

        A "stable" detection is one that has appeared in the world tracker for
        at least ``settle_frames`` consecutive frames with quality at or above
        the locator's ``min_quality``. When multiple matching tracks satisfy
        the gate, the highest-quality one is returned.

        ``settle_frames`` overrides the constructor's value for this call only.
        ``device_index`` selects which Record3D device to open (only matters on
        the first call; subsequent calls reuse the open camera).
        """
        source = self._ensure_source(device_index=device_index)
        settle = int(settle_frames if settle_frames is not None else self._settle_frames)
        target = self._target_label
        deadline = time.monotonic() + float(timeout_s)

        # WorldTracker carries state across locate() calls (intentional — that
        # is where smoothing accrues), but the consecutive-hit gate is per-call
        # so a single locate() request can't return on a track that became
        # stale between calls.
        self._consecutive_hits.clear()
        cached_t_world_camera: Optional[np.ndarray] = None

        while time.monotonic() < deadline:
            packet = source.wait_for_frame(timeout_s=0.25)
            if packet is None:
                continue

            board = detect_board_pose(
                packet.rgb, packet.intrinsic_mat, self._board_cfg
            )
            if (
                board is not None
                and board.reprojection_error_px <= self._anchor_max_reproj_px
            ):
                cached_t_world_camera = np.linalg.inv(board.t_camera_board)

            if cached_t_world_camera is None:
                continue

            detections = self._detector.infer(
                packet.rgb, packet.frame_id, packet.camera_pose
            )
            raw = localize_objects_rgbd(
                packet,
                detections,
                cached_t_world_camera,
                self._localizer_cfg,
                self._profile.table_plane,
            )
            tracked = self._tracker.update(raw)

            seen_ids: set[str] = set()
            ready: Optional[ObjectPoseOutput] = None
            best_quality = -1.0
            for obj in tracked:
                if obj.label != target:
                    continue
                seen_ids.add(obj.object_id)
                self._consecutive_hits[obj.object_id] = (
                    self._consecutive_hits.get(obj.object_id, 0) + 1
                )
                if (
                    self._consecutive_hits[obj.object_id] >= settle
                    and obj.quality >= self._min_quality
                    and obj.quality > best_quality
                ):
                    best_quality = obj.quality
                    ready = obj

            for stale_id in list(self._consecutive_hits.keys()):
                if stale_id not in seen_ids:
                    self._consecutive_hits.pop(stale_id, None)

            if ready is not None:
                return self._to_cup_pose(ready)

        return None

    def _to_cup_pose(self, obj: ObjectPoseOutput) -> CupPose:
        p_world_m = np.asarray(obj.position_world_xyz_m, dtype=np.float64)
        p_robot_m = world_to_robot(p_world_m, self._t_robot_world)
        position_robot_mm = (
            float(p_robot_m[0] * 1000.0),
            float(p_robot_m[1] * 1000.0),
            float(p_robot_m[2] * 1000.0),
        )
        return CupPose(
            position_robot_mm=position_robot_mm,
            position_world_m=(
                float(p_world_m[0]),
                float(p_world_m[1]),
                float(p_world_m[2]),
            ),
            height_m=obj.height_m,
            quality=float(obj.quality),
            yaw_hint_rad=obj.gripper_yaw_hint_rad,
            label=obj.label,
            track_id=obj.object_id,
        )

    def recalibrate(self, **touch_cal_kwargs) -> None:
        """Run the interactive touch-calibration flow, overwrite the profile
        on disk, and reload it into this instance.

        Keyword arguments are forwarded to ``touch_calibrate.main`` via argv,
        with the same flag names (underscores become dashes). The locator's
        ``profile_path`` is always used as the target file.

        Booleans become bare flags: ``legacy_pattern=False`` becomes
        ``--no-legacy-pattern``, ``legacy_pattern=True`` becomes
        ``--legacy-pattern``.
        """
        import sys
        from perception.demos import touch_calibrate as touch_cal_module

        # Release the camera so touch_calibrate can claim it.
        self.close()

        argv = ["touch_calibrate"]
        user_supplied_profile = False
        for key, value in touch_cal_kwargs.items():
            flag_stem = key.replace("_", "-")
            if key == "profile":
                user_supplied_profile = True
            if isinstance(value, bool):
                argv.append(f"--{flag_stem}" if value else f"--no-{flag_stem}")
            else:
                argv.extend([f"--{flag_stem}", str(value)])
        if not user_supplied_profile:
            argv.extend(["--profile", self._profile_path])

        prev_argv = sys.argv
        sys.argv = argv
        try:
            touch_cal_module.main()
        finally:
            sys.argv = prev_argv

        profile = CalibrationProfileIO.load(self._profile_path)
        t_robot_world = profile.get_robot_world_transform()
        if t_robot_world is None or profile.charuco_board is None:
            raise RuntimeError(
                f"recalibration of {self._profile_path!r} produced an incomplete "
                "profile (missing robot_world_transform or charuco_board)."
            )
        self._profile = profile
        self._t_robot_world = np.asarray(t_robot_world, dtype=np.float64).reshape(4, 4)
        self._board_cfg = self._board_cfg_from_spec(profile.charuco_board)
        self._tracker.reset()
        self._consecutive_hits.clear()

    def close(self) -> None:
        if self._source is not None and self._owns_source:
            try:
                self._source.disconnect()
            except Exception:
                pass
        self._source = None
        self._owns_source = False

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return None
