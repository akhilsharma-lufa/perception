"""Forward/inverse kinematics for the myCobot 280 (parallel adaptive gripper).

Why this exists
---------------
The firmware's onboard Cartesian solver (used by `send_coords`) is unreliable on
the 280: it silently rejects targets, picks arbitrary joint branches, and offers
no seeding. This module computes IK ourselves from the URDF so callers can send
deterministic *joint-space* commands (`send_angles`) instead.

The chain is built **explicitly** from the parallel-gripper URDF joint origins
(see `perception/control/models/mycobot_280_jn_adaptive_gripper_parallel.urdf`).
We do NOT use `ikpy.chain.Chain.from_urdf_file` because `gripper_base` has six
child branches plus `mimic` joints, which the auto-parser mishandles.

Two frames, one mapping
-----------------------
- **base frame**: the URDF `g_base` frame (== `joint1`, since `g_base_to_joint1`
  is identity). All FK/IK math happens here.
- **robotcoord frame**: whatever frame pymycobot's `get_coords()` reports the
  flange pose in.

`JointMap` holds the (small) bridge between the two worlds:
  - per-joint `sign` / `offset_deg` between pymycobot `get_angles()` degrees and
    URDF joint radians. Elephant's own MoveIt bridge (`sync_plan.py`) feeds URDF
    radians straight into `send_radians`, so the default is the identity map;
    `ik_debug.py compare` confirms/refines it on hardware.
  - `t_base_robotcoord`: rigid transform so that
    `pose_robotcoord = t_base_robotcoord @ pose_base`. Fitted by `compare`.

The chain ends at the **flange** (`joint6_flange`), which is the frame
`get_coords` reports and the frame `motion_primitives` already targets. The
gripper links + fingertip can be appended for visualization (`include_tip`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ikpy is only needed for the chain/solver, not for JointMap math. Import lazily
# inside the chain builder so JointMap (and load/save) work without ikpy present.


# --- URDF geometry (parallel adaptive gripper) --------------------------------
# Each active joint: (origin_translation_m, origin_rpy_rad, (lower, upper)).
# Axis is z for every joint. Taken verbatim from the vendored URDF.
_HALF_PI = math.pi / 2.0

URDF_ACTIVE_JOINTS = (
    # name,            translation (m),          rpy (rad),                 bounds (rad)
    ("joint2_to_joint1",      (0.0, 0.0, 0.15756), (0.0, 0.0, 0.0),            (-2.9321, 2.9321)),
    ("joint3_to_joint2",      (0.0, 0.0, -0.001),  (0.0, _HALF_PI, -_HALF_PI), (-2.4434, 2.4434)),
    ("joint4_to_joint3",      (-0.1104, 0.0, 0.0), (0.0, 0.0, 0.0),            (-2.6179, 2.6179)),
    ("joint5_to_joint4",      (-0.096, 0.0, 0.06462), (0.0, 0.0, -_HALF_PI),   (-2.6179, 2.6179)),
    ("joint6_to_joint5",      (0.0, -0.07318, 0.0), (_HALF_PI, -_HALF_PI, 0.0), (-2.7052, 2.7925)),
    ("joint6output_to_joint6", (0.0, 0.0456, 0.0), (-_HALF_PI, 0.0, 0.0),       (-3.14, 3.14159)),
)

# Fixed flange->gripper_base, and a nominal fingertip offset down the tool.
_FLANGE_TO_GRIPPER_BASE = ((0.0, 0.0, 0.034), (1.579, 0.0, 0.0))
DEFAULT_TIP_OFFSET_Z_M = 0.111  # caliper-measured AG closed fingertip, matches MotionSettings

N_ACTIVE = 6

_URDF_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mycobot_280_jn_adaptive_gripper_parallel.urdf"
)


def urdf_path() -> Path:
    return _URDF_PATH


# --- Chain construction -------------------------------------------------------

def build_chain(include_tip: bool = False, tip_offset_z_m: float = DEFAULT_TIP_OFFSET_Z_M):
    """Build the ikpy Chain for the 6-DOF arm ending at the flange.

    Links: OriginLink (g_base) + 6 revolute joints -> flange. If `include_tip`,
    appends fixed gripper_base and a fingertip frame `tip_offset_z_m` down the
    tool (visualization only; the flange frame is what `get_coords` reports).

    Returns an `ikpy.chain.Chain`. The full joint vector ikpy expects/returns has
    one entry per link; use `to_full` / `from_full` to convert to/from the 6
    active joint angles.
    """
    from ikpy.chain import Chain  # noqa: PLC0415  (lazy: keeps JointMap import-light)
    from ikpy.link import OriginLink, URDFLink

    links = [OriginLink()]
    mask = [False]
    for name, xyz, rpy, bounds in URDF_ACTIVE_JOINTS:
        links.append(
            URDFLink(
                name=name,
                origin_translation=np.array(xyz, dtype=np.float64),
                origin_orientation=np.array(rpy, dtype=np.float64),
                rotation=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                bounds=bounds,
            )
        )
        mask.append(True)

    if include_tip:
        gb_xyz, gb_rpy = _FLANGE_TO_GRIPPER_BASE
        links.append(
            URDFLink(
                name="gripper_base",
                origin_translation=np.array(gb_xyz, dtype=np.float64),
                origin_orientation=np.array(gb_rpy, dtype=np.float64),
                rotation=None,
                joint_type="fixed",
            )
        )
        mask.append(False)
        links.append(
            URDFLink(
                name="fingertip",
                origin_translation=np.array([0.0, 0.0, float(tip_offset_z_m)], dtype=np.float64),
                origin_orientation=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                rotation=None,
                joint_type="fixed",
            )
        )
        mask.append(False)

    return Chain(links, active_links_mask=mask)


def to_full(chain, active6_rad: Sequence[float]) -> np.ndarray:
    """Expand 6 active joint angles to the full ikpy joint vector (zeros elsewhere)."""
    active6 = np.asarray(active6_rad, dtype=np.float64).reshape(N_ACTIVE)
    full = np.zeros(len(chain.links), dtype=np.float64)
    j = 0
    for i, active in enumerate(chain.active_links_mask):
        if active:
            full[i] = active6[j]
            j += 1
    return full


def from_full(chain, full: Sequence[float]) -> np.ndarray:
    """Extract the 6 active joint angles from a full ikpy joint vector."""
    full = np.asarray(full, dtype=np.float64)
    out = [full[i] for i, active in enumerate(chain.active_links_mask) if active]
    return np.asarray(out, dtype=np.float64)


def fk(chain, active6_rad: Sequence[float]) -> np.ndarray:
    """Forward kinematics: 6 active joint angles (rad) -> 4x4 pose in base frame."""
    return chain.forward_kinematics(to_full(chain, active6_rad))


def ik(
    chain,
    target_xyz_m: Sequence[float],
    target_R: Optional[np.ndarray] = None,
    seed_active6_rad: Optional[Sequence[float]] = None,
    orientation_mode: Optional[str] = None,
) -> np.ndarray:
    """Inverse kinematics in the base frame.

    target_xyz_m   : desired flange position (m) in base frame.
    target_R       : optional 3x3 desired orientation; pair with
                     orientation_mode in {"X","Y","Z","all"}.
    orientation_mode:
        - "all": match the full 3x3 (over-constrains a 6-DOF arm — position can
          drift as the solver favours orientation; avoid for grasping).
        - "Z" (or "X"/"Y"): match only that flange axis to the corresponding
          column of `target_R`. For a top-down grasp, "Z" pins the approach
          axis and leaves redundancy for accurate positioning — preferred.
        - None: position-only.
    seed_active6   : initial guess (rad). Strongly recommended — seeds branch
                     selection (elbow-down) and keeps motion continuous.
    Returns the 6 active joint angles (rad), clamped to joint bounds by ikpy.
    """
    seed_full = (
        to_full(chain, seed_active6_rad)
        if seed_active6_rad is not None
        else None
    )
    kwargs = {"target_position": np.asarray(target_xyz_m, dtype=np.float64).reshape(3)}
    if seed_full is not None:
        kwargs["initial_position"] = seed_full
    if target_R is not None and orientation_mode is not None:
        R = np.asarray(target_R, dtype=np.float64)
        if orientation_mode in ("X", "Y", "Z"):
            # ikpy expects the target axis as a 3-vector for single-axis modes.
            col = {"X": 0, "Y": 1, "Z": 2}[orientation_mode]
            kwargs["target_orientation"] = R[:3, col]
        else:
            kwargs["target_orientation"] = R
        kwargs["orientation_mode"] = orientation_mode
    full = chain.inverse_kinematics(**kwargs)
    return from_full(chain, full)


# --- pymycobot <-> URDF / robotcoord <-> base mapping -------------------------

@dataclass
class JointMap:
    """Bridges pymycobot conventions and the URDF/base frame.

    sign[i], offset_deg[i]: urdf_rad[i] = sign[i]*deg2rad(pymycobot_deg[i]) + deg2rad(offset_deg[i]).
    t_base_robotcoord: 4x4 with pose_robotcoord = t_base_robotcoord @ pose_base.
    Defaults are the identity map (Elephant's MoveIt bridge feeds URDF radians
    straight to send_radians); `ik_debug.py compare` confirms/refines on hardware.
    """

    sign: tuple[float, ...] = (1.0,) * N_ACTIVE
    offset_deg: tuple[float, ...] = (0.0,) * N_ACTIVE
    t_base_robotcoord: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    def pymycobot_deg_to_urdf_rad(self, angles_deg: Sequence[float]) -> np.ndarray:
        a = np.asarray(angles_deg, dtype=np.float64).reshape(N_ACTIVE)
        s = np.asarray(self.sign, dtype=np.float64)
        off = np.radians(np.asarray(self.offset_deg, dtype=np.float64))
        return s * np.radians(a) + off

    def urdf_rad_to_pymycobot_deg(self, angles_rad: Sequence[float]) -> np.ndarray:
        a = np.asarray(angles_rad, dtype=np.float64).reshape(N_ACTIVE)
        s = np.asarray(self.sign, dtype=np.float64)
        off = np.radians(np.asarray(self.offset_deg, dtype=np.float64))
        return np.degrees((a - off) / s)

    def base_point_to_robotcoord(self, p_base_m: Sequence[float]) -> np.ndarray:
        p = np.asarray(p_base_m, dtype=np.float64).reshape(3)
        ph = np.array([p[0], p[1], p[2], 1.0])
        return (self.t_base_robotcoord @ ph)[:3]

    def robotcoord_point_to_base(self, p_robot_m: Sequence[float]) -> np.ndarray:
        p = np.asarray(p_robot_m, dtype=np.float64).reshape(3)
        ph = np.array([p[0], p[1], p[2], 1.0])
        inv = np.linalg.inv(self.t_base_robotcoord)
        return (inv @ ph)[:3]

    # --- persistence ---------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "schema": "perception.kinematics.joint_map.v1",
            "sign": list(self.sign),
            "offset_deg": list(self.offset_deg),
            "t_base_robotcoord": np.asarray(self.t_base_robotcoord, dtype=np.float64).tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "JointMap":
        return cls(
            sign=tuple(float(v) for v in data.get("sign", (1.0,) * N_ACTIVE)),
            offset_deg=tuple(float(v) for v in data.get("offset_deg", (0.0,) * N_ACTIVE)),
            t_base_robotcoord=np.asarray(
                data.get("t_base_robotcoord", np.eye(4).tolist()), dtype=np.float64
            ),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "JointMap":
        return cls.from_json(json.loads(Path(path).read_text()))


DEFAULT_JOINT_MAP_PATH = "calibration/profiles/joint_map.json"


def load_joint_map(path: str | Path = DEFAULT_JOINT_MAP_PATH) -> JointMap:
    """Load a saved JointMap, or return the identity default if none exists."""
    p = Path(path)
    if p.exists():
        return JointMap.load(p)
    return JointMap()
