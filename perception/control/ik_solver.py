"""Runtime IK wrapper: robot-frame flange pose -> pymycobot joint angles.

Holds the loaded ikpy chain + the calibrated `JointMap` so motion primitives can
solve joint angles themselves and command `send_angles` (deterministic) instead
of `send_coords` (firmware Cartesian IK — the unreliable path).

Frame note: callers pass the FLANGE pose in the robot `get_coords` frame (the
same pose `motion_primitives` already computes for `send_coords`). The solver
maps it into the URDF base frame via the JointMap, solves, and maps the angles
back to pymycobot degrees.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from . import kinematics as K


def _rpy_to_R(rpy_deg: Sequence[float]) -> np.ndarray:
    """(Rx,Ry,Rz) deg -> 3x3, XYZ-extrinsic — matches motion_primitives + send_coords."""
    rx, ry, rz = np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rzm @ rym @ rxm


class IKSolveError(RuntimeError):
    """IK produced a solution that does not reach the target within tolerance."""


class IKSolver:
    def __init__(
        self,
        joint_map_path: str = K.DEFAULT_JOINT_MAP_PATH,
        orientation_mode: str = "Z",
        max_pos_err_mm: float = 3.0,
    ):
        self.chain = K.build_chain()
        self.joint_map = K.load_joint_map(joint_map_path)
        self.orientation_mode = orientation_mode
        self.max_pos_err_mm = float(max_pos_err_mm)

    def solve_flange(
        self,
        p_robot_flange_m: Sequence[float],
        rpy_deg: Sequence[float],
        seed_angles_deg: Optional[Sequence[float]] = None,
    ) -> tuple[list[float], float]:
        """Return (pymycobot_angles_deg, position_error_mm) for the flange pose.

        seed_angles_deg: current `get_angles()` — seeds branch selection and keeps
        motion continuous. Falls back to the elbow-down seed if not provided.
        Raises IKSolveError if the round-trip position error exceeds tolerance.
        """
        jm = self.joint_map
        p_base = jm.robotcoord_point_to_base(np.asarray(p_robot_flange_m, dtype=np.float64))

        orient_mode = None if self.orientation_mode == "none" else self.orientation_mode
        R_base = None
        if orient_mode is not None:
            R_base = jm.t_base_robotcoord[:3, :3].T @ _rpy_to_R(rpy_deg)

        # Multi-seed solve: ikpy is seed-sensitive and can settle into a local
        # minimum tens of mm off depending on where it starts. Try the provided
        # seed (current pose, for continuity) plus a spread of fixed configs, and
        # keep the lowest-error solution. This removes the "feasible in the probe
        # but fails on execution" inconsistency (the probe used a different seed).
        seeds_deg: list[np.ndarray] = []
        if seed_angles_deg is not None:
            seeds_deg.append(np.asarray(seed_angles_deg, dtype=np.float64).reshape(6))
        seeds_deg.extend([
            np.array([0.0, -30.0, -30.0, 0.0, 0.0, -45.0]),
            np.array([0.0, -60.0, -20.0, 0.0, 40.0, -45.0]),
            np.array([0.0, -90.0, 20.0, 0.0, 40.0, 0.0]),
            np.array([0.0, -45.0, -45.0, 0.0, 30.0, -45.0]),
            np.array([0.0, -20.0, -60.0, 0.0, 50.0, -45.0]),
            np.array([0.0, -70.0, -10.0, 0.0, 30.0, -90.0]),
        ])

        best_sol = None
        best_err = float("inf")
        for sd in seeds_deg:
            seed_rad = jm.pymycobot_deg_to_urdf_rad(sd)
            sol_rad = K.ik(self.chain, p_base, target_R=R_base,
                           seed_active6_rad=seed_rad, orientation_mode=orient_mode)
            err = float(np.linalg.norm(K.fk(self.chain, sol_rad)[:3, 3] - p_base)) * 1000.0
            if err < best_err:
                best_err = err
                best_sol = sol_rad
            if err <= self.max_pos_err_mm:
                break  # good enough; stop early (first seed tried is the continuous one)

        if best_sol is None or best_err > self.max_pos_err_mm:
            raise IKSolveError(
                f"IK reached only {best_err:.1f} mm from target across "
                f"{len(seeds_deg)} seeds (tol {self.max_pos_err_mm:.1f} mm); "
                f"target likely unreachable/singular."
            )
        return [float(a) for a in jm.urdf_rad_to_pymycobot_deg(best_sol)], best_err
