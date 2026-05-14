from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


DEVICE_TYPE__TRUEDEPTH = 0
DEVICE_TYPE__LIDAR = 1


@dataclass
class FramePacket:
    frame_id: int
    ts_monotonic: float
    rgb: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    intrinsic_mat: np.ndarray
    camera_pose: Any
    device_type: int
    overlays: list["DetectionOverlay"] = field(default_factory=list)


@dataclass
class DetectionOverlay:
    label: str
    confidence: float
    mask: np.ndarray
    anchor_x: int
    anchor_y: int
    distance_m: Optional[float]


@dataclass
class StreamStats:
    frames_enqueued: int = 0
    frames_dropped: int = 0
