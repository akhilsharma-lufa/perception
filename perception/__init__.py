"""Standalone perception library for robot-ready scene understanding."""

from .calibration.multitag_calibrator import (
    AnchorEstimate,
    MultiTagCalibrator,
    MultiTagCalibratorSettings,
    TagDetection,
    TagObservationFrame,
)
from .calibration.profiles import CalibrationProfileIO
from .detection import YoloDetection, YoloDetectorSettings, YoloObjectDetector
from .detection.orientation import RotationCode, detect_orientation, rotate_to_upright
from .io.frame_packet import CameraPose, FramePacket
from .io.record3d_source import Record3DSource
from .localization import RgbdLocalizerSettings, localize_objects_rgbd

__all__ = [
    "AnchorEstimate",
    "CalibrationProfileIO",
    "CameraPose",
    "FramePacket",
    "MultiTagCalibrator",
    "MultiTagCalibratorSettings",
    "Record3DSource",
    "RotationCode",
    "detect_orientation",
    "rotate_to_upright",
    "RgbdLocalizerSettings",
    "TagDetection",
    "TagObservationFrame",
    "YoloDetection",
    "YoloDetectorSettings",
    "YoloObjectDetector",
    "localize_objects_rgbd",
]
