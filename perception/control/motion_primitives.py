"""Top-down grasp motion primitives for the MyCobot 280.

All callers provide world-frame XYZ in meters; primitives convert to robot mm
via the calibrated T_robot_world from the profile and send pymycobot
send_coords commands. RPY is a fixed "gripper pointing down" pose for v1.

Build order matches the plan:
1. world_to_robot  — pure math.
2. project_above_table — translate a world point to a height ABOVE the table.
3. is_reachable    — quick distance check against the arm envelope.
4. move_to_world   — the workhorse: world XYZ + RPY -> send_coords.
5. pre_grasp / descend_and_grasp / lift / place / home — composed primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .gripper import Gripper
from .mycobot_driver import MyCobotDriver


class ReachabilityError(RuntimeError):
    """Raised when a target is outside the arm's reach envelope."""


@dataclass
class MotionSettings:
    # Conservative reach in robot frame, meters. AG-equipped MyCobot 280 has
    # ~37.5 cm absolute tip reach; 27 cm keeps us out of the singular edge.
    max_reach_m: float = 0.27
    # Fixed RPY (deg) interpreted by pymycobot's send_coords as the
    # end-effector orientation. (180, 0, 0) is the canonical "tool0-Z
    # flipped" pose -> gripper points along -robot_Z. Override per call if
    # your robot's Euler convention differs.
    vertical_rpy_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    default_speed: int = 25
    # Composed-primitive defaults — tunable per task.
    hover_height_m: float = 0.08
    lift_height_m: float = 0.10
    grasp_offset_from_table_m: float = 0.025
    grasp_close_value: int = 40
    release_clearance_m: float = 0.05
    # Cartesian send_coords mode: 0 = angular/joint interp (robust — only the
    # endpoint must be reachable, path is whatever the joints can do), 1 =
    # linear cartesian interp (fails silently if any intermediate point on
    # the straight line is unreachable). Default 0 for v1 reliability.
    coord_mode: int = 0
    # Distance from flange (tool0) to the tip of the installed tool. With the
    # canonical "gripper pointing straight down" RPY=(180,0,0), the tip is
    # `tip_offset_z_m` below the flange in robot Z. The calibration step
    # (touch_calibrate.py) records tip positions, so `T_robot_world` maps
    # world -> tip-in-robot; this offset compensates that at runtime so the
    # actual tip lands at the requested world point.
    # 0.125 m = AG gripper flange-to-closed-fingertip. Backed out from a
    # hover=200 mm goto_world test: actual flange robot Z = 184.3 mm, observed
    # fingertip 80 mm above the board (board surface = robot Z ≈ −18.8 mm
    # given the calibration), so true gripper length ≈ 123 mm; we keep 2 mm
    # of headroom above the table by setting the constant to 0.125.
    # Previous tool was a 12 mm pointer (0.012 m).
    tip_offset_z_m: float = 0.125
    # Constant world-frame XY correction (meters) for the gripper centerline.
    # The pointer used during touch_calibrate was glued slightly off-center on
    # the flange, so the calibrated T_robot_world maps `world -> pointer-tip`,
    # but the gripper's actual finger centerline sits offset from where the
    # pointer was. Compensate by SUBTRACTING this from the world target before
    # the transform: command(world) = desired(world) - tip_offset_world_xy_m.
    # Empirically determined for the current setup: with j6 ≈ 0 and RPY=(180,0,0),
    # commanding world (85, 80, 0) lands the centerline at world (100, 80, 0),
    # so the offset is +15 mm in world X (the gripper sits 15 mm to the user's
    # right of where the calibration thinks it is).
    tip_offset_world_xy_m: tuple[float, float] = (0.015, 0.0)


@dataclass
class MotionContext:
    """Bundles the calibration outputs needed by the primitives."""
    t_robot_world: np.ndarray              # 4x4, world -> robot transform
    table_normal_world: np.ndarray         # 3, points "up" (away from table)
    table_origin_world: np.ndarray         # 3, a point on the table plane
    settings: MotionSettings = field(default_factory=MotionSettings)


def world_to_robot(p_world_m: Sequence[float], t_robot_world: np.ndarray) -> np.ndarray:
    """Apply the calibrated transform to a 3-vector in meters."""
    p = np.asarray(p_world_m, dtype=np.float64).reshape(3)
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (np.asarray(t_robot_world, dtype=np.float64) @ ph)[:3]


def project_above_table(
    p_world_m: Sequence[float],
    height_above_table_m: float,
    ctx: MotionContext,
) -> np.ndarray:
    """Return a world-frame point that lies `height_above_table_m` directly above
    the given point's projection onto the table plane.

    Useful for hover/approach poses: regardless of where the cup centroid sits
    in world-Z, this gives you the point you want the gripper tip to occupy
    when hovering above it.
    """
    p = np.asarray(p_world_m, dtype=np.float64).reshape(3)
    n = np.asarray(ctx.table_normal_world, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    origin = np.asarray(ctx.table_origin_world, dtype=np.float64).reshape(3)
    # Drop p to the plane along the normal, then lift to the desired height.
    signed_dist = float(np.dot(p - origin, n))
    on_plane = p - signed_dist * n
    return on_plane + float(height_above_table_m) * n


def is_reachable(p_world_m: Sequence[float], ctx: MotionContext) -> tuple[bool, float]:
    """Returns (reachable, distance_m). Distance is measured in robot frame
    from base origin to the converted point."""
    p_robot = world_to_robot(p_world_m, ctx.t_robot_world)
    dist = float(np.linalg.norm(p_robot))
    return dist <= float(ctx.settings.max_reach_m), dist


def _coords_mm_deg(
    p_robot_m: np.ndarray,
    rpy_deg: tuple[float, float, float],
) -> list[float]:
    return [
        float(p_robot_m[0] * 1000.0),
        float(p_robot_m[1] * 1000.0),
        float(p_robot_m[2] * 1000.0),
        float(rpy_deg[0]),
        float(rpy_deg[1]),
        float(rpy_deg[2]),
    ]


def move_to_world(
    driver: MyCobotDriver,
    p_world_m: Sequence[float],
    ctx: MotionContext,
    speed: Optional[int] = None,
    rpy_deg: Optional[tuple[float, float, float]] = None,
    wait: bool = True,
) -> None:
    # World-XY centerline correction. The calibrated T_robot_world maps to
    # where the touch-calibration pointer was, which is slightly offset from
    # the gripper's actual finger centerline. Shift the world target by the
    # negative offset so the gripper centerline lands at the requested point.
    offset_xy = np.asarray(ctx.settings.tip_offset_world_xy_m, dtype=np.float64)
    p_world_m = np.asarray(p_world_m, dtype=np.float64).reshape(3).copy()
    p_world_m[0] -= float(offset_xy[0])
    p_world_m[1] -= float(offset_xy[1])

    reachable, dist = is_reachable(p_world_m, ctx)
    if not reachable:
        raise ReachabilityError(
            f"world {tuple(round(float(v), 4) for v in p_world_m)} maps to "
            f"{dist*1000:.1f} mm from robot base; max reach is "
            f"{ctx.settings.max_reach_m*1000:.1f} mm"
        )
    # T_robot_world maps world -> tip-in-robot (because touch_calibrate.py
    # records the tip, not the flange). The arm controller takes a FLANGE
    # pose, so push the flange up by tip_offset_z_m in robot Z to land the
    # tip at the commanded world point. Assumes the gripper is pointed down.
    p_robot_tip = world_to_robot(p_world_m, ctx.t_robot_world)
    p_robot_flange = p_robot_tip + np.array(
        [0.0, 0.0, float(ctx.settings.tip_offset_z_m)], dtype=np.float64
    )
    rpy = rpy_deg if rpy_deg is not None else ctx.settings.vertical_rpy_deg
    coords = _coords_mm_deg(p_robot_flange, rpy)
    s = int(speed if speed is not None else ctx.settings.default_speed)
    driver.send_coords_mm_deg(coords, speed=s, mode=int(ctx.settings.coord_mode))
    if wait:
        try:
            driver.wait_until_done(strict=False)
        except Exception:
            # Driver's wait already has a stable-pose fallback; ignore residual issues.
            pass


# ----- Composed primitives -----------------------------------------------------

def pre_grasp(
    driver: MyCobotDriver,
    gripper: Gripper,
    cup_world_m: Sequence[float],
    ctx: MotionContext,
    hover_m: Optional[float] = None,
    speed: Optional[int] = None,
) -> None:
    """Open gripper, then hover above the cup."""
    gripper.open(wait=False)
    hover = float(hover_m if hover_m is not None else ctx.settings.hover_height_m)
    hover_target = project_above_table(cup_world_m, hover, ctx)
    move_to_world(driver, hover_target, ctx, speed=speed)


def descend_and_grasp(
    driver: MyCobotDriver,
    gripper: Gripper,
    cup_world_m: Sequence[float],
    ctx: MotionContext,
    speed: Optional[int] = None,
) -> None:
    """Descend to grasp height above the table and close the gripper."""
    grasp_target = project_above_table(
        cup_world_m, ctx.settings.grasp_offset_from_table_m, ctx
    )
    move_to_world(driver, grasp_target, ctx, speed=speed)
    gripper.close(value=int(ctx.settings.grasp_close_value), wait=True)


def lift(
    driver: MyCobotDriver,
    ctx: MotionContext,
    lift_m: Optional[float] = None,
    speed: Optional[int] = None,
) -> None:
    """Lift relative to current pose by `lift_m` along the table-normal direction.

    Reads the current robot pose, transforms the lift vector from world (via
    table normal) into robot frame, and adds it.
    """
    try:
        coords = driver.get_coords_mm_deg(retries=4)
    except Exception:
        return
    n_world = np.asarray(ctx.table_normal_world, dtype=np.float64).reshape(3)
    n_world = n_world / (np.linalg.norm(n_world) + 1e-12)
    n_robot = ctx.t_robot_world[:3, :3] @ n_world  # direction-only (no translation)
    h = float(lift_m if lift_m is not None else ctx.settings.lift_height_m)
    new_xyz_mm = np.array(coords[:3], dtype=np.float64) + n_robot * (h * 1000.0)
    rpy = coords[3:6]
    s = int(speed if speed is not None else ctx.settings.default_speed)
    driver.send_coords_mm_deg(
        [float(new_xyz_mm[0]), float(new_xyz_mm[1]), float(new_xyz_mm[2]),
         float(rpy[0]), float(rpy[1]), float(rpy[2])],
        speed=s, mode=int(ctx.settings.coord_mode),
    )
    try:
        driver.wait_until_done(strict=False)
    except Exception:
        pass


def place(
    driver: MyCobotDriver,
    gripper: Gripper,
    destination_world_m: Sequence[float],
    ctx: MotionContext,
    release_clearance_m: Optional[float] = None,
    speed: Optional[int] = None,
) -> None:
    """Move above the destination and release the gripper."""
    clearance = float(
        release_clearance_m if release_clearance_m is not None
        else ctx.settings.release_clearance_m
    )
    # Approach from above first.
    above = project_above_table(destination_world_m, ctx.settings.hover_height_m, ctx)
    move_to_world(driver, above, ctx, speed=speed)
    # Descend to release height.
    release_target = project_above_table(destination_world_m, clearance, ctx)
    move_to_world(driver, release_target, ctx, speed=speed)
    gripper.open(wait=True)


def home(
    driver: MyCobotDriver,
    gripper: Optional[Gripper] = None,
    speed: Optional[int] = None,
) -> None:
    """Send the arm to its configured home pose and open the gripper."""
    if gripper is not None:
        gripper.open(wait=False)
    driver.home(speed=speed)
