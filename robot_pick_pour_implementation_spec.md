# iPhone-to-myCobot280 Pick-and-Pour Implementation Spec

This spec defines a practical, deterministic architecture for:

1. locating a source plastic cup with water,
2. picking it using a myCobot280 (6-axis arm),
3. locating a second cup,
4. pouring into the second cup.

Target setup:
- iPhone 16 Pro mounted on a tripod, looking at a table (possibly oblique angle).
- RGB-D stream via `record3d`.
- Robot: myCobot280 with gripper.

---

## 1) Coordinate Frames (Non-Negotiable)

Use these exact frames in code and logs:

- `camera`: iPhone optical frame (depth unprojection output).
- `table`: planar workspace frame (`z=0` on table surface).
- `robot_base`: myCobot base frame.
- `tool0`: end-effector/tool flange frame.
- `cup_src`, `cup_tgt`: object frames centered at detected cup rims or centroids.

Required transforms:

- `T_table_camera`: from camera to table frame (extrinsic from calibration).
- `T_robot_table`: from table to robot base frame (hand-eye/workcell calibration).
- Runtime composition:
  - `T_robot_camera = T_robot_table * T_table_camera`
  - `p_robot = T_robot_camera * p_camera`

If these transforms are inconsistent, everything else fails.

---

## 2) Calibration Procedure

## 2.1 Intrinsics + depth sanity

- Read intrinsics every session from stream (`fx`, `fy`, `tx`, `ty`).
- Validate depth scale with a measured target (e.g., known-height block).
- Acceptance: depth error <= 1.0 cm at working distance.

## 2.2 Table plane calibration

Option A (recommended): place an AprilTag board on table.
- Detect board corners/tags in RGB.
- Use known board geometry to estimate `T_table_camera`.

Option B: depth-plane fit.
- Segment table pixels.
- RANSAC plane fit in camera frame.
- Define `table` origin and axes from plane + chosen anchor point.

Acceptance:
- Reprojection error low and stable across 50+ frames.
- Table normal consistent within small angular jitter.

## 2.3 Robot-to-table calibration

- Collect paired points:
  - Points touched by robot TCP in robot frame.
  - Same points observed in table/camera frame.
- Solve rigid transform (`T_robot_table`) using point-set alignment.

Acceptance:
- Point transform residual <= 1.5 cm in workspace.

---

## 3) Perception Packet Schema

Use one canonical packet object per frame.

```python
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

@dataclass
class FramePacket:
    frame_id: int
    t_capture_monotonic: float
    rgb: np.ndarray               # HxWx3 uint8
    depth: np.ndarray             # HxW float32
    confidence: np.ndarray        # HxW uint8
    K: np.ndarray                 # 3x3 intrinsics
    pose_qt: Dict[str, float]     # qx,qy,qz,qw,tx,ty,tz (camera pose)
    device_type: int              # 0 TrueDepth, 1 LiDAR
    meta: Dict[str, Any]          # fps, dropped frames, latency, etc
```

Rules:
- Never pass raw arrays around independently; always pass `FramePacket`.
- Include frame age and processing latency in `meta`.
- Drop stale packets when overloaded (prefer latest-frame policy).

---

## 4) Object Detection and 3D Localization

## 4.1 Detection model

- Start with YOLO segmentation model (instance masks preferred over bbox-only).
- Classes:
  - `cup_source`
  - `cup_target`

If class split is hard, detect `cup` class and choose source/target by UI click or heuristics.

## 4.2 Depth fusion per object

For each detected cup mask:
- Keep only pixels with high confidence (`confidence >= threshold`).
- Remove outliers (IQR or MAD filter on depth values).
- Use median depth to stabilize range.
- Compute robust 2D centroid from mask.
- Unproject centroid (or multiple mask points) to `p_camera`.

Optionally estimate cup axis by fitting ellipse to rim contour.

Acceptance:
- 3D cup position jitter <= 8 mm over 30 frames (static scene).

---

## 5) Motion Planning for myCobot280

Define these motion primitives:

- `move_pregrasp(p, approach_vec, clearance)`
- `move_grasp(p, orientation)`
- `close_gripper(force_or_width)`
- `lift(delta_z)`
- `move_prepour(p_target, orientation)`
- `tilt_pour(angle, speed, dwell_s)`
- `untilt()`
- `retreat()`

Safety constraints:
- Keep approach vector near table normal for stable grasps.
- Enforce minimum table clearance during horizontal moves.
- Clamp joint velocity and acceleration to conservative limits for liquid handling.

---

## 6) Deterministic State Machine

Use explicit states with timeouts and retry counters.

```text
INIT
  -> CALIBRATE
  -> WAIT_SCENE_STABLE
  -> DETECT_SOURCE
  -> LOCALIZE_SOURCE_3D
  -> PLAN_PICK
  -> EXECUTE_PICK
  -> VERIFY_GRASP
  -> DETECT_TARGET
  -> LOCALIZE_TARGET_3D
  -> PLAN_POUR
  -> EXECUTE_POUR
  -> VERIFY_POUR
  -> PLACE_OR_HOLD
  -> DONE

Any failure -> RECOVER or ABORT_SAFE
```

State output contract:
- Every state emits:
  - `state_name`
  - `entry_time`
  - `exit_reason`
  - metrics snapshot

Recovery examples:
- Detection lost -> rescan from home pose.
- Depth unreliable -> request new frame window and re-estimate.
- Grasp failed -> retry with adjusted grasp point/orientation.

---

## 7) Real-time and Resource Strategy

Since slight slowness is acceptable:
- Run detector at lower rate (e.g., 5-10 Hz), tracking at higher rate.
- Keep robot control loop deterministic and separate from model inference.
- Use latest-frame queue of size 1-2.
- Reduce RGB/depth resolution for perception if needed, preserve calibrated geometry mapping.

---

## 8) Verification and Success Criteria

Primary success criterion:
- Pick source cup, locate target cup, pour into target cup without collision.

Quantitative metrics:
- Cup localization error in robot frame <= 1.5 cm.
- Successful grasp rate >= 90% across 20 trials.
- Pour success (visible transfer to target cup) >= 80% across 20 trials.
- Collision count = 0.

Diagnostics to log:
- Frame latency, drop rate, detector confidence, depth confidence ratio.
- Transform residuals (`camera->table`, `table->robot`).
- State machine transitions and failure reasons.

---

## 9) Recommended Build Order

1. Frame packet + telemetry scaffolding.
2. Calibration pipeline and transform tests.
3. Cup detection + 3D localization.
4. Pick-only state machine.
5. Add target localization and pour sequence.
6. Add retries, fault handling, and trial metrics.

Do not start pouring until pick-only is repeatable.

---

## 10) Minimal File Layout Suggestion

```text
vision/
  perception/
    packet.py
    detect_cups.py
    depth_fusion.py
    transforms.py
  calibration/
    calibrate_table.py
    calibrate_robot_table.py
  control/
    mycobot_driver.py
    motion_primitives.py
    state_machine.py
  app/
    run_pick_pour.py
  logs/
```

This keeps geometry, perception, and robot control cleanly separated.
