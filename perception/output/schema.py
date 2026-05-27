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
    # Upright (optionally tapered) solid model on the table plane, when a table
    # plane is available. `radius_m` is the maximum horizontal radius from the
    # object axis (the rim of a cup — the collision bound); `base_radius_m` is the
    # radius of a low slice near the table. Together they give a linear-cone
    # approximation of radius vs height (equal => cylinder). When a model was fit,
    # `position_world_xyz_m` XY is the de-biased object axis center (not the
    # visible-surface centroid). None when no plane / too few points to fit.
    radius_m: Optional[float] = None
    base_radius_m: Optional[float] = None
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
