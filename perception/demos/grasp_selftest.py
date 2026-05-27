"""Offline self-test for the object model + grasp planner + approach geometry.

No robot, no camera — pure math. Run after tuning gripper/collision numbers to
confirm the perception->plan->orientation->collision path still behaves:

    python -m perception.demos.grasp_selftest

Exits 0 and prints PASS, or raises AssertionError on the first failure.
"""
from __future__ import annotations

import numpy as np

from perception.control.grasp_planner import (
    GraspInfeasible,
    GraspPlannerSettings,
    GripperGeom,
    plan_grasp,
)
from perception.control.motion_primitives import (
    MotionContext,
    MotionSettings,
    _rpy_to_rotation_matrix,
    _rotation_matrix_to_rpy_deg,
    collision_check,
    orientation_rpy_for_approach,
    world_to_robot,
)
from perception.localization.rgbd_localizer import (
    RgbdLocalizerSettings,
    _fit_circle_2d,
    _fit_object_model,
)


def test_circle_fit_from_partial_arc() -> None:
    c = np.array([0.10, 0.08])
    r = 0.025
    ang = np.linspace(-np.pi / 2, np.pi / 2, 40)  # only the camera-facing 180°
    arc = c + r * np.column_stack([np.cos(ang), np.sin(ang)])
    arc += np.random.default_rng(0).normal(0, 5e-4, arc.shape)
    cx, cy, rr = _fit_circle_2d(arc)
    assert np.hypot(cx - c[0], cy - c[1]) < 0.004, (cx, cy)
    assert abs(rr - r) < 0.004, rr


def test_object_model_truncated_cone() -> None:
    c = np.array([0.10, 0.08])
    pts = []
    for h in np.linspace(0, 0.05, 30):
        rad = 0.0175 + (0.025 - 0.0175) * (h / 0.05)
        th = np.linspace(-np.pi / 2, np.pi / 2, 25)
        pts.append(np.column_stack([c[0] + rad * np.cos(th), c[1] + rad * np.sin(th),
                                    np.full_like(th, h)]))
    pts = np.vstack(pts)
    m = _fit_object_model(pts, np.ones(len(pts)), np.array([0., 0., 1.]),
                          np.array([0., 0., 0.]), 0.05, RgbdLocalizerSettings())
    assert m is not None
    axis, radius, base = m
    assert np.hypot(axis[0] - c[0], axis[1] - c[1]) < 0.004
    assert abs(radius - 0.025) < 0.004 and base < radius


def test_shot_cup_needs_angled_grasp() -> None:
    n = np.array([0., 0., 1.]); o = np.zeros(3)
    plan = plan_grasp(np.array([0.1, 0.08, 0.0]), 0.050, 0.025, 0.0175, n, o, GripperGeom())
    assert plan.mode == "angled", plan.mode
    assert plan.grasp_diameter_m <= 0.039 + 1e-6


def test_narrow_cylinder_top_down() -> None:
    n = np.array([0., 0., 1.]); o = np.zeros(3)
    plan = plan_grasp(np.array([0.1, 0.08, 0.0]), 0.060, 0.015, 0.015, n, o, GripperGeom())
    assert plan.mode == "top_down", plan.mode


def test_too_wide_infeasible() -> None:
    n = np.array([0., 0., 1.]); o = np.zeros(3)
    try:
        plan_grasp(np.array([0.1, 0.08, 0.0]), 0.05, 0.030, 0.030, n, o, GripperGeom())
    except GraspInfeasible:
        return
    raise AssertionError("expected GraspInfeasible for an over-wide object")


def test_rpy_roundtrip_and_collision() -> None:
    rng = np.random.default_rng(1)
    for _ in range(500):
        rpy = rng.uniform(-179, 179, 3)
        R = _rpy_to_rotation_matrix(rpy)
        R2 = _rpy_to_rotation_matrix(_rotation_matrix_to_rpy_deg(R))
        assert np.abs(R - R2).max() < 1e-9

    # 180° about X board convention, with translation.
    T = np.eye(4); T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T[:3, 3] = [0.18, -0.06, 0.02]
    ctx = MotionContext(t_robot_world=T, table_normal_world=np.array([0., 0., -1.]),
                        table_origin_world=np.zeros(3), settings=MotionSettings())
    plan = plan_grasp(np.array([0.10, 0.08, -0.025]), 0.05, 0.025, 0.0175,
                      np.array([0., 0., -1.]), np.zeros(3), GripperGeom(),
                      GraspPlannerSettings(approach_tilt_deg=45.0))
    rpy = orientation_rpy_for_approach(plan.approach_dir_world, ctx)
    R = _rpy_to_rotation_matrix(rpy)
    tipoff = np.array([ctx.settings.tip_offset_tool0_xy_m[0],
                       ctx.settings.tip_offset_tool0_xy_m[1], ctx.settings.tip_offset_z_m])
    p_flange = world_to_robot(plan.grasp_point_world_m, T) - R @ tipoff
    ok, clr = collision_check(p_flange, rpy, ctx)
    assert ok and clr > 0.005, (ok, clr)


def main() -> None:
    for fn in (
        test_circle_fit_from_partial_arc,
        test_object_model_truncated_cone,
        test_shot_cup_needs_angled_grasp,
        test_narrow_cylinder_top_down,
        test_too_wide_infeasible,
        test_rpy_roundtrip_and_collision,
    ):
        fn()
        print(f"  ok  {fn.__name__}")
    print("PASS: grasp self-test")


if __name__ == "__main__":
    main()
