from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ObjectPoseOutput:
    object_id: str
    label: str
    position_world_xyz_m: Tuple[float, float, float]
    orientation_world_quat_xyzw: Tuple[float, float, float, float]
    gripper_yaw_hint_rad: Optional[float]
    quality: float
    covariance_diag: Tuple[float, float, float]
    height_m: Optional[float] = None
    source_mode: str = "world"


@dataclass
class PerceptionFrameOutput:
    schema_version: str
    frame_id: int
    ts_monotonic: float
    anchor_mode: str
    anchor_quality: float
    events: list[str] = field(default_factory=list)
    objects: list[ObjectPoseOutput] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)
