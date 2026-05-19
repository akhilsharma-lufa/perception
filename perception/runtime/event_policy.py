from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MotionPolicySettings:
    max_position_delta_m: float = 0.015
    max_orientation_delta_deg: float = 8.0
    min_quality: float = 0.45


@dataclass
class MotionPolicyDecision:
    should_replan: bool
    reason: str


def _rotation_angle_deg(r_prev: np.ndarray, r_new: np.ndarray) -> float:
    r_delta = r_prev.T @ r_new
    trace = float(np.trace(r_delta))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(np.degrees(np.arccos(cos_theta)))


def evaluate_target_motion(
    prev_t_world_object: Optional[np.ndarray],
    new_t_world_object: np.ndarray,
    quality: float,
    settings: MotionPolicySettings,
) -> MotionPolicyDecision:
    if quality < settings.min_quality:
        return MotionPolicyDecision(True, "LOW_CONFIDENCE")
    if prev_t_world_object is None:
        return MotionPolicyDecision(False, "INITIAL")
    pos_delta = float(np.linalg.norm(new_t_world_object[:3, 3] - prev_t_world_object[:3, 3]))
    rot_delta = _rotation_angle_deg(prev_t_world_object[:3, :3], new_t_world_object[:3, :3])
    if pos_delta > settings.max_position_delta_m:
        return MotionPolicyDecision(True, "TARGET_MOVED_REPLAN_REQUIRED")
    if rot_delta > settings.max_orientation_delta_deg:
        return MotionPolicyDecision(True, "TARGET_ROTATED_REPLAN_REQUIRED")
    return MotionPolicyDecision(False, "STABLE")
