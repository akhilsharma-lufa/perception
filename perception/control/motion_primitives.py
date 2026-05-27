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

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .gripper import Gripper
from .mycobot_driver import MyCobotDriver


class ReachabilityError(RuntimeError):
    """Raised when a target is outside the arm's reach envelope."""


class CollisionError(RuntimeError):
    """Raised when a planned flange pose would put the tool collision volume
    below the table plane (e.g. the servo bump dragging on the surface)."""


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
    # 0.111 m = AG gripper flange-to-closed-fingertip (caliper-measured).
    # Note: goto_world.py opens the gripper before moving, so the actual
    # tip Z while running goto_world will be a few mm shorter than 111 mm
    # (fingers swing up slightly when open). For grasping (gripper closes
    # at the descent), 111 mm is the operative value.
    # Previous tool was a 12 mm pointer (0.012 m).
    tip_offset_z_m: float = 0.111
    # Constant TOOL0-FRAME XY offset (meters) for the gripper centerline.
    # The pointer used during touch_calibrate was glued slightly off-center on
    # the flange, so calibration maps `world -> pointer-tip`, but the gripper's
    # actual finger centerline sits offset from where the pointer was. Applied
    # in tool0 frame via the active RPY, so it follows the gripper's
    # orientation:
    #   flange_robot = T @ world - R_robot_tool0 @ (off_x, off_y, tip_offset_z_m)
    # Derivation for the current setup (top-down RPY=(180,0,0)):
    #   user verified that commanding world (85, 80) lands the closed-finger
    #   centerline at world (100, 80) — i.e., the gripper sits +15 mm in
    #   world X compared to where the calibration thinks it is. With
    #   robot X ≈ -world X (from the data) and R for RPY=(180,0,0) being
    #   diag(1, -1, -1), the gripper offset in tool0 frame works out to
    #   (-0.015, 0, 0.125) — negative X in tool0. With this set, commanding
    #   world W lands the centerline at world W exactly.
    tip_offset_tool0_xy_m: tuple[float, float] = (-0.015, 0.0)
    # When True, compute IK ourselves (perception.control.ik_solver) and command
    # send_angles instead of send_coords. This bypasses the firmware's unreliable
    # Cartesian solver (silent rejections, arbitrary elbow/wrist branches) and is
    # the path to repeatable motion. Requires a fitted JointMap
    # (ik_debug.py compare -> calibration/profiles/joint_map.json). Default False
    # preserves the legacy send_coords behaviour until validated on hardware.
    use_ik_solver: bool = False
    # Orientation constraint for the IK path: "Z" pins the approach axis (correct
    # for top-down grasps, accurate positioning), "all" matches full orientation,
    # "none" is position-only. See ik_solver / kinematics.ik.
    ik_orientation_mode: str = "Z"
    ik_max_pos_err_mm: float = 3.0
    # --- Tool collision volume (tool0 frame, meters) ---
    # Boxes approximating the gripper so non-top-down approaches don't drag the
    # servo bump into the table. tool0 axes: +Z = approach (flange -> fingertips),
    # +X = finger-open axis, +Y = servo-bump side (perpendicular to finger plane).
    # Each entry is (center_xyz, half_extents_xyz). Defaults from the AG photos
    # (~110x90x60 mm body + a servo dome protruding ~+Y, slightly off-center in X);
    # MEASURE and tune on hardware before trusting an angled grasp.
    tool_collision_boxes: tuple = (
        ((0.0, 0.0, 0.055), (0.045, 0.030, 0.055)),     # gripper body
        ((0.008, 0.045, 0.028), (0.024, 0.024, 0.032)),  # servo bump (+Y, off-center)
    )
    # Minimum clearance (m) every collision-box corner must keep above the table.
    table_clearance_m: float = 0.005
    # Gate moves on the collision check. Off by default (top-down demos never
    # collide); the angled-grasp path turns it on.
    enable_collision_check: bool = False


@dataclass
class MotionContext:
    """Bundles the calibration outputs needed by the primitives."""
    t_robot_world: np.ndarray              # 4x4, world -> robot transform
    table_normal_world: np.ndarray         # 3, points "up" (away from table)
    table_origin_world: np.ndarray         # 3, a point on the table plane
    settings: MotionSettings = field(default_factory=MotionSettings)
    # Lazily-built IK solver (only when settings.use_ik_solver). Kept on the
    # context so the chain + JointMap load once and are reused across primitives.
    ik_solver: object = None

    def get_ik_solver(self):
        """Return the IKSolver, building it on first use. Import is lazy so the
        ikpy dependency is only required when the IK path is enabled."""
        if self.ik_solver is None:
            from .ik_solver import IKSolver
            self.ik_solver = IKSolver(
                orientation_mode=self.settings.ik_orientation_mode,
                max_pos_err_mm=self.settings.ik_max_pos_err_mm,
            )
        return self.ik_solver


def world_to_robot(p_world_m: Sequence[float], t_robot_world: np.ndarray) -> np.ndarray:
    """Apply the calibrated transform to a 3-vector in meters."""
    p = np.asarray(p_world_m, dtype=np.float64).reshape(3)
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (np.asarray(t_robot_world, dtype=np.float64) @ ph)[:3]


def _rpy_to_rotation_matrix(rpy_deg: Sequence[float]) -> np.ndarray:
    """Convert (Rx, Ry, Rz) Euler angles in degrees to a 3x3 rotation matrix.

    Convention: XYZ extrinsic (i.e., R = Rz @ Ry @ Rx). This is consistent
    with pymycobot's send_coords RPY interpretation. With RPY=(180, 0, 0),
    the resulting matrix has tool0 +Z aligned with robot -Z (gripper points
    down), which matches what we've verified empirically.
    """
    rx, ry, rz = np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))
    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_mat @ ry_mat @ rx_mat


def _rotation_matrix_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `_rpy_to_rotation_matrix` (R = Rz @ Ry @ Rx). Returns (rx,ry,rz) deg."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    cy = float(np.sqrt(max(0.0, 1.0 - sy * sy)))
    if cy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(sy, cy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: pitch ~ +-90 deg, fix roll = 0
        rx = 0.0
        ry = np.arctan2(sy, cy)
        rz = np.arctan2(-R[0, 1], R[1, 1])
    return (float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz)))


def orientation_rpy_for_approach(
    approach_dir_world: Sequence[float],
    ctx: MotionContext,
) -> tuple[float, float, float]:
    """Build an end-effector RPY (deg, robot frame) for an angled approach.

    Aligns tool +Z with the approach direction (the way the gripper travels into
    the object) and rolls so tool +Y (the servo-bump side) points as far UP as
    possible, keeping the bump clear of the table. The world approach vector is
    rotated into the robot frame via the calibrated transform.
    """
    r_rw = np.asarray(ctx.t_robot_world, dtype=np.float64)[:3, :3]
    z = r_rw @ np.asarray(approach_dir_world, dtype=np.float64).reshape(3)
    z = z / (np.linalg.norm(z) + 1e-12)
    up = r_rw @ (np.asarray(ctx.table_normal_world, dtype=np.float64).reshape(3))
    up = up / (np.linalg.norm(up) + 1e-12)
    y = up - np.dot(up, z) * z
    if np.linalg.norm(y) < 1e-6:  # approach is ~parallel to up; pick any perpendicular
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y = ref - np.dot(ref, z) * z
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)  # re-orthonormalize, right-handed (z = x cross y)
    R = np.column_stack([x, y, z])
    return _rotation_matrix_to_rpy_deg(R)


def _tool_box_corners(center: Sequence[float], half: Sequence[float]) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = np.asarray(half, dtype=np.float64).reshape(3)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    return c + signs * h


def collision_check(
    p_robot_flange_m: np.ndarray,
    rpy_deg: Sequence[float],
    ctx: MotionContext,
) -> tuple[bool, float]:
    """Check the tool collision volume against the table plane for a flange pose.

    Transforms every collision-box corner (tool0 -> robot) and measures its signed
    distance above the table plane (in robot frame). Returns (ok, min_clearance_m);
    ok is False if any corner is closer to the table than `settings.table_clearance_m`.
    """
    boxes = ctx.settings.tool_collision_boxes
    if not boxes:
        return True, float("inf")
    p = np.asarray(p_robot_flange_m, dtype=np.float64).reshape(3)
    R = _rpy_to_rotation_matrix(rpy_deg)
    r_rw = np.asarray(ctx.t_robot_world, dtype=np.float64)[:3, :3]
    n_robot = r_rw @ np.asarray(ctx.table_normal_world, dtype=np.float64).reshape(3)
    n_robot = n_robot / (np.linalg.norm(n_robot) + 1e-12)
    origin_robot = world_to_robot(ctx.table_origin_world, ctx.t_robot_world)
    min_clear = float("inf")
    for center, half in boxes:
        corners_tool0 = _tool_box_corners(center, half)
        corners_robot = (R @ corners_tool0.T).T + p
        signed = (corners_robot - origin_robot) @ n_robot
        min_clear = min(min_clear, float(np.min(signed)))
    return (min_clear >= float(ctx.settings.table_clearance_m)), min_clear


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


def _command_flange(
    driver: MyCobotDriver,
    p_robot_flange_m: np.ndarray,
    rpy_deg: tuple[float, float, float],
    ctx: MotionContext,
    speed: Optional[int],
    wait: bool,
) -> None:
    """Send the arm to a flange pose, via our IK + send_angles when
    `settings.use_ik_solver`, else via the legacy send_coords path."""
    s = int(speed if speed is not None else ctx.settings.default_speed)
    if ctx.settings.use_ik_solver:
        solver = ctx.get_ik_solver()
        try:
            seed = driver.get_angles_deg(retries=3)
        except Exception:
            seed = None
        angles_deg, _err = solver.solve_flange(p_robot_flange_m, rpy_deg, seed_angles_deg=seed)
        driver.send_angles_deg(angles_deg, speed=s)
    else:
        driver.send_coords_mm_deg(
            _coords_mm_deg(p_robot_flange_m, rpy_deg),
            speed=s, mode=int(ctx.settings.coord_mode),
        )
    if wait:
        try:
            driver.wait_until_done(strict=False)
        except Exception:
            # Driver's wait already has a stable-pose fallback; ignore residual issues.
            pass


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
    p_world_m = np.asarray(p_world_m, dtype=np.float64).reshape(3).copy()
    reachable, dist = is_reachable(p_world_m, ctx)
    if not reachable:
        raise ReachabilityError(
            f"world {tuple(round(float(v), 4) for v in p_world_m)} maps to "
            f"{dist*1000:.1f} mm from robot base; max reach is "
            f"{ctx.settings.max_reach_m*1000:.1f} mm"
        )
    # T_robot_world maps world -> tip-in-robot (because touch_calibrate.py
    # records the tip, not the flange). The arm controller takes a FLANGE
    # pose, so subtract the tool0 +Z direction (in robot frame) times the
    # tip-offset length to get from tip back to flange. This makes the
    # offset RPY-aware: for top-down (RPY=180,0,0), tool0 +Z = -robot_Z so
    # subtracting it bumps the flange UP in robot Z (matches previous code);
    # for side approach (e.g., RPY=0,-90,0), tool0 +Z is horizontal so the
    # flange is offset horizontally, not vertically.
    rpy = rpy_deg if rpy_deg is not None else ctx.settings.vertical_rpy_deg
    p_robot_tip = world_to_robot(p_world_m, ctx.t_robot_world)
    r_robot_tool0 = _rpy_to_rotation_matrix(rpy)
    # Full 3D tool0-frame offset from flange to gripper centerline tip:
    #   (off_x, off_y, off_z) — XY is the centerline lateral correction
    #   from the touch-calibration pointer's mounting offset; Z is the
    #   tool length (flange face to fingertip).
    off_x, off_y = ctx.settings.tip_offset_tool0_xy_m
    tip_offset_tool0 = np.array(
        [float(off_x), float(off_y), float(ctx.settings.tip_offset_z_m)],
        dtype=np.float64,
    )
    p_robot_flange = p_robot_tip - r_robot_tool0 @ tip_offset_tool0
    if ctx.settings.enable_collision_check:
        ok, clearance = collision_check(p_robot_flange, rpy, ctx)
        if not ok:
            raise CollisionError(
                f"flange pose at world {tuple(round(float(v),3) for v in p_world_m)} "
                f"rpy {tuple(round(float(v),1) for v in rpy)} puts the tool "
                f"{clearance*1000:.1f} mm from the table (need "
                f"{ctx.settings.table_clearance_m*1000:.1f} mm) — would collide."
            )
    _command_flange(driver, p_robot_flange, rpy, ctx, speed=speed, wait=wait)


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
    rpy = (float(coords[3]), float(coords[4]), float(coords[5]))
    # coords/new_xyz_mm are the flange pose in the robot frame already (lift is a
    # relative move read back from get_coords), so command it directly.
    _command_flange(driver, new_xyz_mm * 1e-3, rpy, ctx, speed=speed, wait=True)


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


def approach_and_grasp(
    driver: MyCobotDriver,
    gripper: Gripper,
    plan,
    ctx: MotionContext,
    speed: Optional[int] = None,
    approach_speed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Execute a grasp from a `GraspPlan` (grasp_planner.GraspPlan; duck-typed).

    Opens the gripper, moves to the pre-grasp standoff with the plan's approach
    orientation (tool +Z along the approach, servo bump up), advances along the
    approach axis to the grasp point, then closes. Both moves are collision-gated
    when `settings.enable_collision_check`. Returns the RPY used (deg).
    """
    rpy = orientation_rpy_for_approach(plan.approach_dir_world, ctx)
    gripper.open(wait=True)
    move_to_world(driver, plan.pregrasp_point_world_m, ctx, speed=speed, rpy_deg=rpy)
    move_to_world(driver, plan.grasp_point_world_m, ctx,
                  speed=(approach_speed if approach_speed is not None else speed), rpy_deg=rpy)
    gripper.close(value=int(ctx.settings.grasp_close_value), wait=True)
    return rpy


def retreat_along_approach(
    driver: MyCobotDriver,
    plan,
    ctx: MotionContext,
    dist_m: Optional[float] = None,
    speed: Optional[int] = None,
) -> None:
    """Back the grasped object straight out along the (reverse) approach axis —
    up and away from the table — before transit, keeping the same orientation."""
    rpy = orientation_rpy_for_approach(plan.approach_dir_world, ctx)
    d = float(dist_m if dist_m is not None else ctx.settings.lift_height_m)
    retreat_point = np.asarray(plan.grasp_point_world_m, dtype=np.float64) \
        - np.asarray(plan.approach_dir_world, dtype=np.float64) * d
    move_to_world(driver, retreat_point, ctx, speed=speed, rpy_deg=rpy)


def home(
    driver: MyCobotDriver,
    gripper: Optional[Gripper] = None,
    speed: Optional[int] = None,
) -> None:
    """Send the arm to its configured home pose and open the gripper."""
    if gripper is not None:
        gripper.open(wait=False)
    driver.home(speed=speed)


def safe_home(
    driver: MyCobotDriver,
    gripper: Optional[Gripper] = None,
    *,
    speed: int = 30,
    timeout_s: float = 15.0,
    tol_deg: float = 3.0,
    open_gripper: bool = True,
) -> bool:
    """Reliably park the arm at [0,0,0,0,0,0] and open the gripper.

    Recovery-grade homing: works from any pose (including a stuck/awkward one),
    re-energises the servos first in case a prior run released them, drops
    whatever is held, then verifies arrival **by joint angle** rather than the
    firmware's ``is_moving()`` flag (which on this 280 firmware reports 0 before
    the motion even starts — see PICK_PLACE.md "Home reliability").

    Returns True if all six joints reach 0 within ``tol_deg`` before
    ``timeout_s``, else False. Never raises for a non-arrival; it logs the final
    angles and returns the boolean so the caller/CLI decides what to do. Leaves
    the servos engaged.
    """
    HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 1. Re-energise — the arm may be limp (released) or stuck mid-fault.
    try:
        driver.power_on()
        time.sleep(0.5)
    except Exception as exc:
        print(f"[safe_home] WARN: power_on failed: {exc}")

    # 2. Drop whatever is held before the arm sweeps home.
    if gripper is not None and open_gripper:
        try:
            gripper.open(wait=True)
        except Exception as exc:
            print(f"[safe_home] WARN: gripper open failed: {exc}")

    # 3. Diagnostics: where are we starting from?
    try:
        start = driver.get_angles_deg(retries=3)
        print(f"[safe_home] from angles: {[round(a, 1) for a in start]}")
    except Exception:
        print("[safe_home] could not read starting angles")

    # 4. Command home (joint space — deterministic).
    driver.send_angles_deg(HOME, speed=int(speed))

    # 5. Verify by joint angle, polling until converged or timed out.
    deadline = time.monotonic() + float(timeout_s)
    last: Optional[Sequence[float]] = None
    while time.monotonic() < deadline:
        try:
            last = driver.get_angles_deg(retries=2)
        except Exception:
            time.sleep(0.15)
            continue
        if max(abs(float(last[i])) for i in range(6)) <= float(tol_deg):
            print(f"[safe_home] HOME OK: {[round(float(a), 1) for a in last]}")
            return True
        time.sleep(0.15)

    print(
        f"[safe_home] HOME FAILED: joints still off after {timeout_s:.0f}s: "
        f"{[round(float(a), 1) for a in last] if last is not None else 'unknown'}"
    )
    return False
