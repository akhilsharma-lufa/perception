from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    qx: float
    qy: float
    qz: float
    qw: float
    tx: float
    ty: float
    tz: float

    @classmethod
    def from_record3d(cls, pose: Any) -> "CameraPose":
        return cls(
            qx=float(pose.qx),
            qy=float(pose.qy),
            qz=float(pose.qz),
            qw=float(pose.qw),
            tx=float(pose.tx),
            ty=float(pose.ty),
            tz=float(pose.tz),
        )


@dataclass
class FramePacket:
    frame_id: int
    ts_monotonic: float
    rgb: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    intrinsic_mat: np.ndarray  # by source convention, normalized to RGB resolution
    camera_pose: CameraPose
    device_type: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    world_camera: Optional[np.ndarray] = None
    # (H, W) the intrinsic matrix is calibrated at. Producers should emit
    # `intrinsic_mat` already scaled to RGB resolution (the convention downstream
    # code assumes); this field carries the calibration shape for traceability.
    intrinsic_shape: Optional[tuple] = None
