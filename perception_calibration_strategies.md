# Perception Calibration Strategies (Automated)

This document defines calibration strategies for the new `perception` library with:

- iPhone 16 Pro (Record3D USB stream),
- `tag36h11` AprilTags,
- `tag 1` fixed as world origin,
- optional `tag 2` and `tag 3` (or more) for stability.

---

## 1) Strategy Summary

### Strategy A: Single-tag anchor (minimum viable)

- Use only `tag 1` as origin.
- Pros: simplest setup.
- Cons: higher jitter, fragile under occlusion, weaker angle conditioning.
- Use only for quick prototyping.

### Strategy B: Boardless multi-tag anchor (recommended)

- Keep `tag 1` as origin.
- Add free-placed support tags (`tag 2`, `tag 3`, ...).
- Build session tag-constellation map automatically.
- Fuse all visible mapped tags each frame.
- Pros: high stability, robust to partial occlusions, portable scene.

### Strategy C: Multi-tag + table-plane validation (high confidence)

- Strategy B plus depth-based table plane consistency checks.
- Detects scene disturbances and poor calibration sessions earlier.
- Best for robot-facing runs.

---

## 2) Physical Setup SOP (No fixed board required)

1. Fix iPhone on tripod; avoid hand-held camera.
2. Place `tag 1` near the center of the active workspace.
3. Place `tag 2`, `tag 3` in a wide non-collinear triangle around workspace.
4. Keep at least 3 tags visible in normal operation.
5. Ensure tags are flat, rigid, and not curved.
6. Maintain good lighting and avoid strong reflections.

Recommended geometry:

- `tag1 -> tag2` and `tag1 -> tag3`: `0.20m` to `0.40m`.
- Angle at `tag1`: roughly `60` to `120` degrees.

---

## 3) Automated Calibration Pipeline

Implemented in:

- `perception/calibration/multitag_calibrator.py`
- `perception/calibration/automation.py`
- demo: `perception/demos/auto_calibrate_tags.py`

### Step A: Session bootstrap

- Collect `N` frames (default `~120`) of tag observations.
- For each frame, estimate `T_camera_tag_i` for each visible tag.
- Require origin tag (`tag 1`) to be observed enough times.

### Step B: Build world map

- Define world frame as origin tag frame (`T_world_tag1 = I`).
- Estimate `T_world_tag_i` for each auxiliary tag using robust averaging.
- Persist profile JSON with metrics.

### Step C: Runtime anchoring

- On each frame:
  - compute `T_world_camera` candidates from all visible mapped tags,
  - fuse candidates into one estimate,
  - compute residual metrics and quality score.
- Emit anchor events on health changes.

---

## 4) Runtime Events and Safety Implications

- `ANCHOR_HOLD_LAST`: no mapped tags now, holding recent estimate for short window.
- `ANCHOR_LOST`: cannot anchor world reliably; robot-facing updates should be frozen.
- `ANCHOR_UNSTABLE`: mapped tags disagree strongly; pose quality degraded.
- `TARGET_MOVED_REPLAN_REQUIRED`: object moved significantly; trigger replanning.

Robot integration guidance:

- Never execute a new grasp/pour command from `ANCHOR_LOST`.
- Require minimum quality threshold for closed-loop motion updates.

---

## 5) Transform Math Reference

Core equations:

- `T_world_object = T_world_camera * T_camera_object`
- `T_robot_object = T_robot_world * T_world_object`

Where:

- `T_world_camera` comes from multi-tag anchor solve,
- `T_camera_object` comes from depth + segmentation localization,
- `T_robot_world` is solved after robot arrives.

All transforms are 4x4 homogeneous matrices.

---

## 6) Camera Pose (`get_camera_pose`) Usage

Use Record3D camera pose as secondary signal:

- short-term smoothing,
- camera movement detection,
- latency-aware interpolation.

Do not use it as sole absolute world anchor for robot actions.

---

## 7) Supporting Many Object Types

Per-class adapter model:

- common detector interface,
- class-specific localization and orientation policies,
- class-specific confidence and grasp hints.

Examples:

- cup: rim/axis + pour orientation hint,
- bottle: long axis + cap orientation,
- box: dominant face normals.

This keeps core perception generic and scalable.

---

## 8) Recalibration Triggers (Automated)

Trigger re-bootstrap when:

- mean anchor residual exceeds threshold for sustained window,
- no origin visibility for prolonged periods,
- observed inter-tag geometry drifts beyond expected tolerance.

Suggested thresholds (initial):

- anchor translation RMSE > `3cm`,
- anchor rotation RMSE > `4deg`,
- inter-tag geometry drift > `2cm` mean.

---

## 9) Operational Checklist

Before run:

1. Run `python -m perception.demos.auto_calibrate_tags`.
2. Confirm profile saved and includes `tag 1` and auxiliary tags.
3. Verify stable anchor quality in static scene.
4. Run `python -m perception.demos.live_monitor --profile calibration/profiles/session_multitag.json`.
5. Check portrait/landscape toggles (`p`, `l`, `k`, `o`) and confirm overlays remain coherent.

During run:

1. Watch anchor event stream.
2. Pause motion if `ANCHOR_LOST` or repeated `ANCHOR_UNSTABLE`.
3. Re-bootstrap if geometry drift event occurs.

After robot delivery:

1. Solve and save `T_robot_world`.
2. Validate end-to-end object localization error in robot frame.
