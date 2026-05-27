"""Debug the myCobot 280 inverse kinematics.

Our own IK (ikpy, built from the URDF) replaces the firmware's unreliable
Cartesian solver: we solve joint angles ourselves and command `send_angles`.
This tool validates that solver and — critically — locks the mapping between
pymycobot's conventions and the URDF/base frame (`compare`), without which our
FK won't match the real robot.

Offline subcommands (no robot, headless-safe): fk, ik, roundtrip, plot.
Robot subcommands (need pymycobot + a connected arm): compare, solve-and-send.

Examples
--------
    # Forward kinematics for a joint vector (pymycobot degrees):
    python -m perception.demos.ik_debug fk --angles 0 -30 -30 0 0 -45

    # Inverse kinematics for a base-frame target (mm), top-down:
    python -m perception.demos.ik_debug ik --xyz 200 0 150 --rpy 180 0 0

    # Solver self-consistency over random reachable poses:
    python -m perception.demos.ik_debug roundtrip --n 1000

    # Lock the pymycobot<->URDF mapping on hardware (run once after a setup change):
    python -m perception.demos.ik_debug compare --port /dev/ttyUSB0 --samples 12

    # Solve IK for a get_coords-frame target and actually move there (guarded):
    python -m perception.demos.ik_debug solve-and-send --xyz 200 0 150 --rpy 180 0 0 --yes
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional, Sequence

import numpy as np

from perception.control import kinematics as K


# --- small shared helpers -----------------------------------------------------

def _rpy_to_R(rpy_deg: Sequence[float]) -> np.ndarray:
    """(Rx,Ry,Rz) degrees -> 3x3, XYZ-extrinsic (R = Rz@Ry@Rx), matching
    motion_primitives._rpy_to_rotation_matrix / pymycobot send_coords."""
    rx, ry, rz = np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rzm @ rym @ rxm


def _R_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of _rpy_to_R (XYZ-extrinsic)."""
    R = np.asarray(R, dtype=np.float64)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def _fmt_mm(v) -> str:
    return "(" + ", ".join(f"{x*1000:+8.1f}" for x in v[:3]) + ") mm"


def _fmt_deg(v) -> str:
    return "[" + ", ".join(f"{x:+7.2f}" for x in v) + "]"


def _check_limits(active6_rad: np.ndarray) -> list[str]:
    warnings = []
    for i, (name, _xyz, _rpy, (lo, hi)) in enumerate(K.URDF_ACTIVE_JOINTS):
        a = float(active6_rad[i])
        margin = min(a - lo, hi - a)
        if a < lo or a > hi:
            warnings.append(f"  J{i+1} {math.degrees(a):+.1f}deg OUT OF LIMIT "
                            f"[{math.degrees(lo):+.0f},{math.degrees(hi):+.0f}]")
        elif margin < math.radians(5.0):
            warnings.append(f"  J{i+1} {math.degrees(a):+.1f}deg near limit "
                            f"(margin {math.degrees(margin):.1f}deg)")
    return warnings


# Seed that biases an elbow-down, shoulder-forward branch (matches goto_cup.py).
ELBOW_DOWN_SEED_DEG = (0.0, -30.0, -30.0, 0.0, 0.0, -45.0)


# --- subcommand: fk -----------------------------------------------------------

def cmd_fk(args) -> int:
    jm = K.load_joint_map(args.joint_map)
    ch = K.build_chain(include_tip=args.tip)
    angles_deg = np.asarray(args.angles, dtype=np.float64)
    urdf_rad = jm.pymycobot_deg_to_urdf_rad(angles_deg)
    T_base = K.fk(ch, urdf_rad)
    p_robot = jm.base_point_to_robotcoord(T_base[:3, 3])
    R_robot = jm.t_base_robotcoord[:3, :3] @ T_base[:3, :3]
    print(f"input pymycobot angles (deg): {_fmt_deg(angles_deg)}")
    print(f"urdf joint values     (deg): {_fmt_deg(np.degrees(urdf_rad))}")
    print(f"flange in BASE frame      : {_fmt_mm(T_base[:3, 3])}  "
          f"rpy={tuple(round(v,1) for v in _R_to_rpy_deg(T_base[:3,:3]))}")
    print(f"flange in ROBOTCOORD frame: {_fmt_mm(p_robot)}  "
          f"rpy={tuple(round(v,1) for v in _R_to_rpy_deg(R_robot))}")
    if args.tip:
        print("  (chain includes gripper_base + fingertip; pose above is the TIP)")
    return 0


# --- subcommand: ik -----------------------------------------------------------

def cmd_ik(args) -> int:
    jm = K.load_joint_map(args.joint_map)
    ch = K.build_chain()
    # Target given in robotcoord frame unless --base-frame.
    target_robot_m = np.asarray(args.xyz, dtype=np.float64) * 1e-3
    target_base_m = (target_robot_m if args.base_frame
                     else jm.robotcoord_point_to_base(target_robot_m))

    R_target = None
    orient_mode = None
    if args.rpy is not None:
        R_robot = _rpy_to_R(args.rpy)
        R_target = (R_robot if args.base_frame
                    else jm.t_base_robotcoord[:3, :3].T @ R_robot)
        orient_mode = None if args.orient == "none" else args.orient

    seed_deg = np.asarray(args.seed if args.seed is not None else ELBOW_DOWN_SEED_DEG,
                          dtype=np.float64)
    seed_rad = jm.pymycobot_deg_to_urdf_rad(seed_deg)

    sol_rad = K.ik(ch, target_base_m, target_R=R_target,
                   seed_active6_rad=seed_rad, orientation_mode=orient_mode)
    sol_deg = jm.urdf_rad_to_pymycobot_deg(sol_rad)

    # round-trip error
    T = K.fk(ch, sol_rad)
    pos_err_mm = float(np.linalg.norm(T[:3, 3] - target_base_m)) * 1000.0

    print(f"target ({'base' if args.base_frame else 'robotcoord'} frame): "
          f"{_fmt_mm(target_base_m if args.base_frame else target_robot_m)}"
          + (f"  rpy={tuple(args.rpy)}" if args.rpy is not None else ""))
    print(f"IK solution pymycobot angles (deg): {_fmt_deg(sol_deg)}")
    print(f"FK round-trip position error: {pos_err_mm:.3f} mm")
    warns = _check_limits(sol_rad)
    if warns:
        print("joint-limit notes:")
        print("\n".join(warns))
    if pos_err_mm > 2.0:
        print("WARN: large round-trip error — target may be unreachable or a "
              "singular/edge configuration. Try a different seed or rpy.")
    return 0


# --- subcommand: roundtrip ----------------------------------------------------

def cmd_roundtrip(args) -> int:
    """FK->IK->FK over random reachable poses, mimicking real usage: the IK is
    seeded from a small perturbation of the true config (as it is at runtime,
    where we seed from the current joint angles) and constrains the approach
    axis only (--mode Z), which is what a top-down grasp needs.

    Pass --far-seed to stress-test branch recovery from an arbitrary seed; that
    regime is far harder and NOT representative of how the solver is used.
    """
    rng = np.random.default_rng(args.seed_rng)
    ch = K.build_chain()
    lows = np.array([lo for _n, _t, _r, (lo, _hi) in K.URDF_ACTIVE_JOINTS])
    highs = np.array([hi for _n, _t, _r, (_lo, hi) in K.URDF_ACTIVE_JOINTS])
    # Sample inside 85% of each joint range to stay off the hard stops.
    mid = 0.5 * (lows + highs)
    span = 0.85 * 0.5 * (highs - lows)
    mode = None if args.mode == "none" else args.mode
    seed_jitter = math.radians(args.seed_jitter_deg)

    pos_errs, z_errs, fails = [], [], 0
    for _ in range(args.n):
        q = mid + span * (2.0 * rng.random(K.N_ACTIVE) - 1.0)
        T = K.fk(ch, q)
        if args.far_seed:
            seed = mid + span * (2.0 * rng.random(K.N_ACTIVE) - 1.0)
        else:
            seed = q + seed_jitter * (2.0 * rng.random(K.N_ACTIVE) - 1.0)
        sol = K.ik(ch, T[:3, 3], target_R=(T[:3, :3] if mode else None),
                   seed_active6_rad=seed, orientation_mode=mode)
        Ts = K.fk(ch, sol)
        pe = float(np.linalg.norm(Ts[:3, 3] - T[:3, 3])) * 1000.0
        # approach-axis (flange Z) error in degrees
        cos = abs(float(np.dot(T[:3, 2], Ts[:3, 2])))
        ze = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        pos_errs.append(pe)
        z_errs.append(ze)
        if pe > 1.0:
            fails += 1

    pos = np.asarray(pos_errs)
    zer = np.asarray(z_errs)
    print(f"roundtrip over {args.n} poses (mode={args.mode}, "
          f"{'FAR seed' if args.far_seed else f'near seed ±{args.seed_jitter_deg:.0f}deg'}):")
    for label, arr, unit in (("position", pos, "mm"), ("approach-axis", zer, "deg")):
        print(f"  {label:13s} median={np.median(arr):.3f} p90={np.percentile(arr,90):.3f} "
              f"max={arr.max():.3f} {unit}")
    print(f"  solutions with >1mm position error: {fails}/{args.n} "
          f"({100.0*fails/args.n:.1f}%)")
    return 0 if fails < max(1, args.n // 20) else 1


# --- subcommand: plot ---------------------------------------------------------

def cmd_plot(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jm = K.load_joint_map(args.joint_map)
    ch = K.build_chain(include_tip=args.tip)
    urdf_rad = jm.pymycobot_deg_to_urdf_rad(np.asarray(args.angles, dtype=np.float64))
    full = K.to_full(ch, urdf_rad)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ch.plot(full, ax)
    ax.set_title("myCobot 280 — angles(deg) "
                 + "[" + ", ".join(f"{float(a):.0f}" for a in args.angles) + "]")
    fig.savefig(args.save, dpi=120)
    print(f"saved {args.save}")
    return 0


# --- robot helpers ------------------------------------------------------------

def _connect_driver(args):
    from perception.control import MyCobotDriver, MyCobotDriverSettings
    drv = MyCobotDriver(MyCobotDriverSettings(port=args.port, baudrate=args.baudrate))
    drv.connect()
    drv.power_on()
    time.sleep(0.5)
    return drv


# --- subcommand: compare (fit & save JointMap) --------------------------------

def cmd_compare(args) -> int:
    from perception.calibration.robot_calibrator import kabsch_align

    jm = K.load_joint_map(args.joint_map)  # use current sign/offset for FK
    ch = K.build_chain()
    drv = _connect_driver(args)
    base_pts: list[np.ndarray] = []
    robot_pts: list[np.ndarray] = []
    try:
        if args.manual:
            print("MANUAL mode: releasing servos. Move the arm by hand to varied "
                  "poses; press Enter to capture each (q + Enter to finish).")
            drv.release_all_servos()
            while True:
                s = input(f"[{len(base_pts)} captured] Enter=capture, q=done: ").strip()
                if s.lower() == "q":
                    break
                _capture_sample(drv, jm, ch, base_pts, robot_pts)
        else:
            print(f"AUTO mode: sampling {args.samples} poses around the elbow-down "
                  f"seed at speed {args.speed}. Ensure the arm has clear space.")
            rng = np.random.default_rng(0)
            seed = np.asarray(ELBOW_DOWN_SEED_DEG, dtype=np.float64)
            for k in range(args.samples):
                jitter = (2.0 * rng.random(6) - 1.0) * np.array([25, 20, 20, 25, 25, 30])
                target = np.clip(seed + jitter, -120, 120)
                drv.send_angles_deg(list(target), speed=args.speed)
                drv.wait_until_done(strict=False)
                time.sleep(0.4)
                _capture_sample(drv, jm, ch, base_pts, robot_pts)
                print(f"  captured {k+1}/{args.samples}")
    finally:
        try:
            drv.disconnect()
        except Exception:
            pass

    if len(base_pts) < 3:
        print(f"ERROR: need >=3 samples, got {len(base_pts)}.", file=sys.stderr)
        return 2

    res = kabsch_align(base_pts, robot_pts)  # maps base -> robotcoord
    jm.t_base_robotcoord = res.t_robot_world
    print(f"\nfitted t_base_robotcoord (base -> get_coords frame), "
          f"RMSE = {res.rmse_m*1000:.2f} mm over {len(base_pts)} samples")
    print(np.array_str(res.t_base_robotcoord, precision=4, suppress_small=True))
    print(f"per-sample residual mm: {np.round(res.per_point_residual_m*1000,2).tolist()}")
    if res.rmse_m * 1000 > 8.0:
        print("WARN: RMSE > 8 mm. The identity pymycobot<->URDF sign/offset map "
              "may be wrong for your firmware; inspect per-sample residuals and "
              "consider per-joint sign flips before trusting IK.")
    if args.save:
        jm.save(args.joint_map)
        print(f"saved JointMap -> {args.joint_map}")
    else:
        print("(dry: pass --save to persist the JointMap)")
    return 0


def _capture_sample(drv, jm, ch, base_pts, robot_pts) -> None:
    angles = drv.get_angles_deg(retries=5)
    coords = drv.get_coords_mm_deg(retries=5)
    urdf_rad = jm.pymycobot_deg_to_urdf_rad(angles)
    p_base = K.fk(ch, urdf_rad)[:3, 3]
    base_pts.append(np.asarray(p_base, dtype=np.float64))
    robot_pts.append(np.asarray(coords[:3], dtype=np.float64) * 1e-3)


# --- subcommand: solve-and-send -----------------------------------------------

def cmd_solve_and_send(args) -> int:
    jm = K.load_joint_map(args.joint_map)
    ch = K.build_chain()
    target_robot_m = np.asarray(args.xyz, dtype=np.float64) * 1e-3
    target_base_m = jm.robotcoord_point_to_base(target_robot_m)
    R_target = orient_mode = None
    if args.rpy is not None:
        R_target = jm.t_base_robotcoord[:3, :3].T @ _rpy_to_R(args.rpy)
        orient_mode = None if args.orient == "none" else args.orient

    drv = _connect_driver(args)
    try:
        cur = drv.get_angles_deg(retries=5)
        seed_rad = jm.pymycobot_deg_to_urdf_rad(cur)
        sol_rad = K.ik(ch, target_base_m, target_R=R_target,
                       seed_active6_rad=seed_rad, orientation_mode=orient_mode)
        sol_deg = jm.urdf_rad_to_pymycobot_deg(sol_rad)
        pos_err_mm = float(np.linalg.norm(K.fk(ch, sol_rad)[:3, 3] - target_base_m)) * 1000

        print(f"current angles (deg): {_fmt_deg(cur)}")
        print(f"IK solution    (deg): {_fmt_deg(sol_deg)}")
        print(f"FK round-trip error : {pos_err_mm:.3f} mm")
        warns = _check_limits(sol_rad)
        if warns:
            print("joint-limit notes:\n" + "\n".join(warns))
        if pos_err_mm > 3.0:
            print("ABORT: round-trip error too large; refusing to send.", file=sys.stderr)
            return 3
        if not args.yes:
            print("(dry: pass --yes to actually send these angles)")
            return 0
        drv.send_angles_deg(list(sol_deg), speed=args.speed)
        drv.wait_until_done(strict=False)
        time.sleep(0.3)
        ach = drv.get_coords_mm_deg(retries=5)
        err = float(np.linalg.norm(np.array(ach[:3]) - target_robot_m * 1000))
        print(f"achieved get_coords: {_fmt_mm(np.array(ach[:3])/1000)}  "
              f"-> error vs target {err:.1f} mm")
    finally:
        try:
            drv.disconnect()
        except Exception:
            pass
    return 0


# --- arg parsing --------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m perception.demos.ik_debug",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint-map", default=K.DEFAULT_JOINT_MAP_PATH,
                   help="Path to joint_map.json (identity default if absent).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fk", help="forward kinematics for a joint vector")
    s.add_argument("--angles", type=float, nargs=6, required=True,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                   help="pymycobot joint angles in degrees")
    s.add_argument("--tip", action="store_true", help="report fingertip instead of flange")
    s.set_defaults(func=cmd_fk)

    s = sub.add_parser("ik", help="inverse kinematics for a Cartesian target")
    s.add_argument("--xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                   help="target position in mm")
    s.add_argument("--rpy", type=float, nargs=3, default=None, metavar=("RX", "RY", "RZ"),
                   help="target orientation in degrees (default: position-only)")
    s.add_argument("--seed", type=float, nargs=6, default=None,
                   help="seed joint angles (deg); default = elbow-down seed")
    s.add_argument("--orient", choices=("none", "Z", "all"), default="Z",
                   help="orientation constraint when --rpy given (default Z = approach axis)")
    s.add_argument("--base-frame", action="store_true",
                   help="interpret --xyz/--rpy in the URDF base frame (default: get_coords frame)")
    s.set_defaults(func=cmd_ik)

    s = sub.add_parser("roundtrip", help="FK->IK->FK self-consistency over random poses")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed-rng", type=int, default=0)
    s.add_argument("--mode", choices=("none", "Z", "all"), default="Z",
                   help="orientation constraint (default Z = approach axis, grasp-relevant)")
    s.add_argument("--seed-jitter-deg", type=float, default=8.0,
                   help="near-seed perturbation magnitude (deg)")
    s.add_argument("--far-seed", action="store_true",
                   help="seed from an arbitrary config (adversarial; not representative)")
    s.set_defaults(func=cmd_roundtrip)

    s = sub.add_parser("plot", help="render a 3D stick figure to PNG (headless)")
    s.add_argument("--angles", type=float, nargs=6, required=True,
                   help="pymycobot joint angles in degrees")
    s.add_argument("--tip", action="store_true")
    s.add_argument("--save", default="ik_pose.png")
    s.set_defaults(func=cmd_plot)

    # robot-connected
    for name, func, help_ in (("compare", cmd_compare, "fit & save pymycobot<->URDF JointMap"),
                              ("solve-and-send", cmd_solve_and_send, "solve IK and move the arm")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--port", default="/dev/ttyUSB0")
        s.add_argument("--baudrate", type=int, default=1_000_000)
        s.add_argument("--speed", type=int, default=20)
        if name == "compare":
            s.add_argument("--samples", type=int, default=12,
                           help="auto-mode pose count")
            s.add_argument("--manual", action="store_true",
                           help="hand-guide the arm and capture on Enter instead of auto-sampling")
            s.add_argument("--save", action="store_true", help="persist the fitted JointMap")
        else:
            s.add_argument("--xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                           help="target position in mm (get_coords frame)")
            s.add_argument("--rpy", type=float, nargs=3, default=None,
                           metavar=("RX", "RY", "RZ"), help="target orientation in degrees")
            s.add_argument("--orient", choices=("none", "Z", "all"), default="Z",
                           help="orientation constraint when --rpy given (default Z = approach axis)")
            s.add_argument("--yes", action="store_true", help="actually send (else dry-run)")
        s.set_defaults(func=func)

    return p


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
