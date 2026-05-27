"""Grasp planning: turn a perceived object model into a grasp + approach pose.

Pure geometry (no robot/ikpy imports) so it is unit-testable offline. Consumes the
object model from the perception layer (axis center on the table plane, height, max
radius = rim, base-slice radius) and the gripper's finger-gap limits, and returns
where to grasp and how to approach.

The motivating case is the shot cup: a truncated cone with rim Ø 50 mm (wider than the
gripper's 45 mm max opening) and base Ø 35 mm. The open fingers cannot clear the rim
from straight above, so the cup must be grasped on its lower body via an angled
approach. This planner finds the highest still-graspable slice (most finger contact
while the open gap still fits) and the approach vector to reach it.

Frames: everything is world-frame meters. The table plane is given by a unit normal
(pointing up, away from the table) and an origin point on the plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


class GraspInfeasible(RuntimeError):
    """No graspable slice exists (object wider than the gripper everywhere, or too
    thin to hold), so the caller must not attempt a pick."""


@dataclass
class GripperGeom:
    """Parallel-finger gap limits, meters. Defaults match the AG: 20-45 mm gap."""
    min_gap_m: float = 0.020
    max_gap_m: float = 0.045
    # Total clearance subtracted from max_gap so the open fingers actually clear the
    # object diameter (finger thickness + positional error + a safety band).
    grip_clearance_m: float = 0.006


@dataclass
class GraspPlannerSettings:
    # Tilt of the approach axis from vertical for the angled (non-top-down) grasp.
    # 0 deg = straight down; 90 deg = horizontal side approach.
    approach_tilt_deg: float = 45.0
    # Horizontal heading (rad) the approach comes FROM, in the table plane. The tool
    # travels toward the object along -heading while descending. Pick a heading whose
    # side of the object is clear of obstacles and keeps the servo bump up. Default
    # along world +X; the motion layer / operator should set the clear side.
    approach_yaw_rad: float = 0.0
    # How far back along the approach axis the pre-grasp standoff sits.
    standoff_m: float = 0.06
    # Keep the grasp slice at least this far above the table (finger/table clearance).
    min_grasp_height_m: float = 0.008
    # Back off the solved highest-graspable slice by this much so we sit safely inside
    # the graspable band rather than right at the limit.
    grasp_height_safety_m: float = 0.004
    # If the widest part of the object already fits the open gripper, a clean vertical
    # (top-down) grasp is used instead of the angled approach.
    prefer_top_down_when_fits: bool = True


@dataclass
class GraspPlan:
    mode: str                          # "top_down" | "angled"
    grasp_point_world_m: np.ndarray    # finger-midpoint target = object axis @ grasp height
    approach_dir_world: np.ndarray     # unit vector the tool travels along INTO the grasp
    pregrasp_point_world_m: np.ndarray # standoff start, back along the approach axis
    grasp_height_m: float              # height above table of the grasp slice
    grasp_diameter_m: float            # object diameter at the grasp slice
    tilt_deg: float                    # approach tilt from vertical actually used
    notes: str = ""


def _diameter_at_height(h: float, height_m: float, base_radius_m: float, radius_m: float) -> float:
    """Linear-cone diameter at height ``h`` above the table.

    Assumes radius varies linearly from ``base_radius_m`` at the table to ``radius_m``
    (the max/rim radius) at the top. For a cylinder the two radii are equal.
    """
    if height_m <= 1e-6:
        return 2.0 * radius_m
    frac = float(np.clip(h / height_m, 0.0, 1.0))
    r = base_radius_m + (radius_m - base_radius_m) * frac
    return 2.0 * float(r)


def plan_grasp(
    axis_center_world_m: np.ndarray,
    height_m: float,
    radius_m: float,
    base_radius_m: Optional[float],
    plane_normal_world: np.ndarray,
    plane_origin_world: np.ndarray,
    gripper: Optional[GripperGeom] = None,
    settings: Optional[GraspPlannerSettings] = None,
) -> GraspPlan:
    """Plan a grasp for an upright (optionally tapered) object on the table plane.

    Raises ``GraspInfeasible`` if no slice fits the gripper, or if the graspable
    slice is thinner than the gripper can close onto.
    """
    g = gripper or GripperGeom()
    s = settings or GraspPlannerSettings()
    base_r = float(base_radius_m if base_radius_m is not None else radius_m)
    rim_r = float(radius_m)
    H = float(max(height_m, 0.0))

    n = np.asarray(plane_normal_world, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    axis = np.asarray(axis_center_world_m, dtype=np.float64).reshape(3)

    effective_max = float(g.max_gap_m - g.grip_clearance_m)

    # Narrowest diameter is at the base (for a cup widening upward). If even that
    # doesn't fit the open gripper, there's nowhere to grasp.
    narrow_d = _diameter_at_height(0.0, H, base_r, rim_r)
    wide_d = _diameter_at_height(H, H, base_r, rim_r)
    min_d = min(narrow_d, wide_d)
    max_d = max(narrow_d, wide_d)
    if min_d > effective_max:
        raise GraspInfeasible(
            f"object min diameter {min_d*1000:.1f} mm exceeds usable gripper opening "
            f"{effective_max*1000:.1f} mm (max {g.max_gap_m*1000:.0f} mm - "
            f"{g.grip_clearance_m*1000:.0f} mm clearance); cannot grasp."
        )

    in_plane_basis = _plane_basis(n)
    top_down_fits = max_d <= effective_max and s.prefer_top_down_when_fits

    if top_down_fits:
        # Whole object fits the open fingers from above — clean vertical grasp.
        grasp_h = float(np.clip(0.5 * H, s.min_grasp_height_m, max(H, s.min_grasp_height_m)))
        approach_dir = -n  # straight down
        tilt = 0.0
        mode = "top_down"
    else:
        # Find the highest slice that still fits, then back off into the band.
        grasp_h = _highest_graspable_height(H, base_r, rim_r, effective_max)
        grasp_h = float(np.clip(
            grasp_h - s.grasp_height_safety_m, s.min_grasp_height_m, max(H, s.min_grasp_height_m)
        ))
        approach_dir = _angled_approach_dir(n, in_plane_basis, s.approach_yaw_rad, s.approach_tilt_deg)
        tilt = float(s.approach_tilt_deg)
        mode = "angled"

    grasp_d = _diameter_at_height(grasp_h, H, base_r, rim_r)
    notes = ""
    if grasp_d < g.min_gap_m:
        notes = (f"grasp slice Ø {grasp_d*1000:.1f} mm is below the gripper's min gap "
                 f"{g.min_gap_m*1000:.0f} mm; fingers may bottom out before gripping firmly.")

    grasp_point = axis + grasp_h * n
    pregrasp_point = grasp_point - approach_dir * float(s.standoff_m)

    return GraspPlan(
        mode=mode,
        grasp_point_world_m=grasp_point,
        approach_dir_world=approach_dir,
        pregrasp_point_world_m=pregrasp_point,
        grasp_height_m=grasp_h,
        grasp_diameter_m=float(grasp_d),
        tilt_deg=tilt,
        notes=notes,
    )


def _highest_graspable_height(H: float, base_r: float, rim_r: float, effective_max: float) -> float:
    """Height of the highest slice whose diameter == effective_max (linear cone).

    For a cup widening upward this is where the body becomes too wide; grasping just
    below it maximizes finger contact while the open gap still clears the object.
    """
    if H <= 1e-6 or abs(rim_r - base_r) < 1e-9:
        return 0.5 * H  # cylinder: any height works; pick mid-body
    # 2*(base_r + (rim_r-base_r)*(h/H)) = effective_max
    h = H * (effective_max / 2.0 - base_r) / (rim_r - base_r)
    return float(np.clip(h, 0.0, H))


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    return u, v


def _angled_approach_dir(
    normal: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray],
    yaw_rad: float,
    tilt_deg: float,
) -> np.ndarray:
    """Unit vector the tool travels along to reach the grasp.

    tilt=0 -> straight down (-normal); tilt=90 -> horizontal, coming from the
    ``yaw`` heading side. The tool sits up-and-to-the-heading-side at standoff and
    advances along this vector into the object.
    """
    u, v = basis
    up = np.asarray(normal, dtype=np.float64).reshape(3)
    up = up / (np.linalg.norm(up) + 1e-12)
    hdir = math.cos(yaw_rad) * u + math.sin(yaw_rad) * v
    tilt = math.radians(float(tilt_deg))
    d = -(math.cos(tilt) * up + math.sin(tilt) * hdir)
    return d / (np.linalg.norm(d) + 1e-12)
