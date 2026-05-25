# Pick-and-Place — Session Log & Operator Guide

This document covers the pick-and-place demo for the MyCobot 280 + AG (Adaptive
Gripper) on the Jetson Orin NX. It's both a *reference* (how to run the demo,
what each flag does, what's expected to happen) and a *session log* (what
broke, why, how we fixed it, what's still open).

If you're just running the demo, jump to [Running the demo](#running-the-demo).
The rest of the document explains *why* each step exists.

---

## Hardware setup

- **Arm:** MyCobot 280 (6-DOF), on `/dev/ttyUSB0`, baud 1 000 000.
- **End-effector:** MyCobot Adaptive Gripper (AG). 20–45 mm finger gap.
  ~110 × 90 × 60 mm body, with a **servo housing protruding ~2 cm from
  one face** (the "servo bump"). At joint-zero (`[0,0,0,0,0,0]`) the
  fingers are parallel to the floor and the servo bump faces world-up.
- **Camera:** iPhone 16 Pro on a fixed tripod, Record3D over USB to the
  Jetson. iPhone Auto-Lock must be set to **Never** or the depth stream
  drops out.
- **Fiducial:** ChArUco board on the table — 11 × 8 squares, 20 mm
  square, 14 mm marker, DICT_4X4_50, **legacy pattern (top-left = tag)**.
- **World frame:** origin at the board's top-left inner chessboard
  corner. X along the long edge, Y along the short edge. The `+Z` axis
  points *into the table* (i.e. "above the table" in world is
  `Z < 0`). This sign convention is baked into the saved
  `table_normal_world = (0, 0, -1)`.
- **Calibration profile:** `calibration/profiles/session_multitag.json`.
  Achieved 2.54 mm Kabsch RMSE via `touch_calibrate.py`.
- **Tip-offset (flange face → closed fingertip in tool0 +Z):** 0.125 m.
  Backed out empirically from a `goto_world --hover-mm 200` run that
  measured the fingertip 80 mm above the board.

The arm sits at the edge of the working surface. **Workspace is on one
side of the base only** — monitors / obstacles occupy the other side.
J1 sweeps centered around +90° to stay in the safe arc.

---

## Running the demo

The recommended sequence is:

1. **Gripper smoke test** (verify open/close):
   ```
   python3 -c "from perception.control.mycobot_driver import MyCobotDriver, MyCobotDriverSettings; from perception.control.gripper import Gripper, GripperSettings; d=MyCobotDriver(MyCobotDriverSettings(port='/dev/ttyUSB0')); d.connect(); d.power_on(); g=Gripper(d, GripperSettings()); g.open(); g.close(value=40); g.open()"
   ```
   Watch the fingers physically: should go open → partly close → open.
   The polarity inversion is configured in `GripperSettings.invert_polarity=True`;
   if your gripper unit obeys the documented "0=open, 100=closed"
   convention, flip that to `False`.

2. **Quick reach + alignment verification** (no cup):
   Place a coin / piece of tape at the inner corner at world (100, 80, 0)
   — 5 squares right, 4 squares down from the top-left corner where 4
   squares meet. Then:
   ```
   python3 -m perception.demos.goto_world \
     --world-mm 100 80 0 --hover-mm 30 --rpy 180 0 0 \
     --speed 15 --port /dev/ttyUSB0 --max-reach-mm 320
   ```
   During the 2 s hold, sight straight down between the closed fingertips.
   Midpoint should land on the marker. If it's off by more than a few mm,
   adjust `MotionSettings.tip_offset_tool0_xy_m` (see
   [Centerline correction](#centerline-correction) below).

3. **YOLO sanity check** (without arm):
   ```
   python3 -m perception.demos.yolo_world_monitor \
     --classes "cup,bottle,wine glass,bowl,vase" \
     --min-confidence 0.25
   ```
   Confirm your cup is consistently masked. Note the label YOLO assigns —
   for a small red shot glass it may sometimes be `wine glass` or `bowl`
   instead of `cup`. Remember the dominant label.

4. **Detection-only dry-run of pick-place**:
   ```
   python3 -m perception.demos.pick_place_cup --port /dev/ttyUSB0 \
     --classes "cup,bottle,wine glass,bowl,vase" \
     --target-label cup \
     --min-confidence 0.25 \
     --detection-window-s 4.0 \
     --place-mm 100 80 0 \
     --grasp-close-value 70 \
     --xy-bias-mm 11 7 \
     --hover-mm 30 \
     --dry-run
   ```
   Confirms: cup is detected, world XYZ is sensible (~100, 80, ≈ −25 mm
   for a 5 cm cup), reach passes for both pick and place.

5. **Live pick-place** (drop `--dry-run`):
   ```
   python3 -m perception.demos.pick_place_cup --port /dev/ttyUSB0 \
     --classes "cup,bottle,wine glass,bowl,vase" \
     --target-label cup \
     --min-confidence 0.25 \
     --detection-window-s 4.0 \
     --place-mm 40 40 0 \
     --grasp-close-value 70 \
     --xy-bias-mm 11 7 \
     --hover-mm 30
   ```
   Expected: pre_grasp (hover above cup) → descend_and_grasp (close on cup
   walls; AG's adaptive mechanism stalls at first contact) → lift →
   place at world (40, 40, 0) → release → back off → home.
   Cup should physically end up at the (40, 40) inner corner.

### CLI flag reference (pick_place_cup.py)

| Flag | Default | Purpose |
|---|---|---|
| `--profile` | `calibration/profiles/session_multitag.json` | Calibration profile. |
| `--port` | `/dev/ttyUSB0` | Arm serial port. |
| `--classes` | `cup` | Comma-separated YOLO labels to ACCEPT (filter). Use `*` to disable filtering. |
| `--target-label` | `cup` | Of the accepted labels, which one to actually pick. |
| `--place-mm` | `0 0 0` | Destination XYZ in world mm. |
| `--hover-mm` | `50` | Transport altitude above the table. Lower this to 30 if the flange reach budget is tight (see [Reach envelope](#reach-envelope)). |
| `--release-mm` | `30` | Altitude at which the gripper opens at the destination. |
| `--grasp-close-value` | `40` | Public-convention 0..100. With the AG's adaptive mechanism this is the *target* — the gripper will stall at first contact, usually well before the commanded value. 70 ≈ "try to close tight." |
| `--detection-window-s` | `2.0` | How long to observe before moving. 4 s gives more frames to average over. |
| `--min-confidence` | `0.5` | Below this YOLO confidence, detections are dropped. 0.25 matches the monitor. |
| `--xy-bias-mm` | `0 0` | Constant XY offset (mm) added to the detected cup world XY to compensate for YOLO centroid bias. Tuned to (+11, +7) for the current setup. |
| `--max-reach-mm` | `320` | Tip-reach gate. 320 accommodates the AG-equipped flange's wider envelope. |
| `--speed` | `20` | Cartesian motion speed (1–100). Stay slow for safety. |
| `--dry-run` | off | Detect + print plan; do NOT move the arm. |

---

## Why each piece exists — full session history

This section is the long-form story of every issue we hit and how we
resolved it, in roughly the order it came up. Each entry is independent
— skim or skip.

### Tip offset for the new tool (a.k.a. "how long is the gripper?")

The calibration profile was originally fit using a 12 mm pointer tool. With
the AG installed, the tool is much longer; if we don't tell the code, the
arm will plant the gripper into the table. We back the new length out by:

- Running `goto_world --world-mm 0 0 0 --hover-mm 200 --rpy 180 0 0`.
  Because we haven't changed the code yet, the arm goes to the same FLANGE
  position the pointer-calibrated arm would have. The flange ends up
  212 mm above the table.
- Measuring the gap from board to *gripper* fingertip. Observed: 80 mm.
- Gripper length = 212 − 80 = **132 mm**. We set
  `MotionSettings.tip_offset_z_m = 0.125` (≈ 7 mm of upward safety
  margin: the tip lands a hair above the commanded Z rather than below).

This is hardcoded; if you swap to a different tool, redo this measurement.

### Reach envelope

The MyCobot 280's joints are limited; with a 125 mm tool extending in
tool0 +Z, the *flange* must sit further from the base than the *tip*.
For a top-down grasp at world (0, 0, 0), the flange ends up at a robot
norm of ~289 mm — past the AG-equipped arm's ~280 mm soft physical
limit. Symptoms when this happens: pre_grasp's `move_to_world` is
silently rejected by IK and the arm jumps straight from prepose to the
descend pose, skipping the controlled hover.

Mitigations:

- Use the **board center** (~100, 70, 0) as your test target, not the
  origin (0, 0, 0). The origin is at the far corner from the arm and
  hits the reach edge.
- Lower `--hover-mm` to 30 (default is 50). The lower the hover, the
  smaller the flange norm.
- Bump `--max-reach-mm` to 320 (default is 270). This widens the
  software gate but doesn't override the arm's physical limit.

### Centerline correction

`touch_calibrate` recorded positions of the *pointer tip*, which was
glued to the flange face. Empirically, the pointer was mounted slightly
off-center (~15 mm in tool0 +X direction): when we now command world
(100, 80, 0), the AG's actual finger centerline lands at world (115, 80,
0). We verified this by commanding world (85, 80, 0) and seeing the
centerline land exactly on the marker at (100, 80, 0).

**Fix:** add a tool0-frame XY offset to `MotionSettings`:

```
tip_offset_tool0_xy_m = (-0.015, 0.0)
```

Inside `move_to_world`:
```
flange_robot = T_robot_world @ world − R_robot_tool0 @ (off_x, off_y, tip_offset_z_m)
```

The offset is applied in **tool0 frame**, transformed via the active
RPY's rotation matrix. This means the same offset correctly handles any
gripper orientation (top-down, side, angled) — for top-down (RPY=180,0,0)
it collapses to a 15 mm world-X shift, which matches the empirical
verification.

If you swap the gripper or remount the pointer, redo this verification.

### Centroid bias (YOLO localization)

The RGB-D localizer (`localize_objects_rgbd`) returns the 3D centroid of
all visible mask pixels. YOLO's mask only covers the **camera-facing
half** of the cup, so the centroid is biased toward the camera. With
the iPhone on the tripod and the cup near board center, the bias is
about +11 mm in world X and +7 mm in world Y — i.e., YOLO localizes the
cup ~13 mm toward the camera from its true axis.

**Workaround (current):** pass `--xy-bias-mm 11 7` to shift the detected
position back to the true cup axis before the arm goes there.

**Proper fixes (future):**
- Custom YOLO training on the actual cup (30–50 annotated images) would
  give much better masks and possibly bias closer to zero.
- Fit a circle/cylinder to the cup mask in 2D and compute the
  intersection with the table plane — that gives the cup *base* center,
  which is unbiased.

### Gripper polarity (the big one)

The MyCobot AG firmware on this unit interprets the value parameter
**inverted from the documented convention**:

- Documented: `value=0` → fully open, `value=100` → fully closed.
- Actual on this hardware: `value=0` → CLOSED, `value=100` → OPEN.

This was the single biggest source of confusion in the session. With
the inverted polarity but the code thinking conventional polarity:

- `gripper.open()` (sends `value=0`) actually **closed** the gripper.
- `gripper.close(value=70)` (sends `value=70`) actually drove the
  gripper toward open.
- During descent, the fingers were closer together than expected →
  hit the cup top instead of going around it.
- During lift, the cup got "gripped" by an unintended inside-out
  mechanism (fingers expanding inside the cup mouth).
- At place, `gripper.open()` (sends `value=0`) actually closed the
  gripper, releasing the cup by *retracting* the inside-out pressure.

**Fix:** added `invert_polarity: bool = True` to `GripperSettings`.
Inside `Gripper.set_width()` and `Gripper.get_value()`, the value is
flipped at the API boundary:

```
wire_v = 100 - v   if invert_polarity else v
```

All higher-level code continues to use the documented "0=open,
100=closed" convention. Every existing CLI flag (`--grasp-close-value`
etc.) keeps its semantics.

**To verify polarity on a new gripper:** run the smoke test (step 1 in
"Running the demo"). The fingers should physically go: spread → partly
close → spread. If they go: close → partly open → close, flip
`invert_polarity` to `False`.

### YOLO detection reliability

`yolo_world_monitor` and `pick_place_cup` use different YOLO settings.
The monitor is permissive (`--min-confidence 0.25`, broad class
whitelist, 12-frame hold). The pick demos default to strict (`0.5`,
only `cup`, no hold). Consequence: in the live workflow, the monitor
shows a clear cup but the pick demo says "no detection."

**Recommended flags for the pick demo:**
- `--classes "cup,bottle,wine glass,bowl,vase"` — accept any cup-like label
- `--target-label cup` — but only pick the `cup` one (or change to
  whichever label dominates for your cup in the monitor)
- `--min-confidence 0.25` — match the monitor
- `--detection-window-s 4.0` — double the window so noisy frames don't
  dominate

This won't fix the underlying small-cup-at-an-angle problem, but it
makes the pick demo as detection-reliable as the monitor.

### Home reliability

`driver.home()` calls `send_angles_deg(home_angles_deg)` and then
`wait_until_done()`. On this firmware, `is_moving()` sometimes returns
0 *before* the motion has actually started, causing `wait_until_done`
to return immediately. The subsequent `get_coords_mm_deg` then reads
the pre-motion pose, the script exits, and the arm hasn't actually gone
home.

**Fix in `pick_place_cup.py`:** before issuing the home command, pause
1.5 s for any residual back-off motion to finish; after the home
command, sleep 5 s before reading the pose. This empirically gives the
joints enough time to swing to home angles.

If your firmware behaves differently (`is_moving()` works reliably),
this delay just adds wall-clock time without changing correctness.

### The side-approach experiment (deferred)

We attempted a side-approach FSM (`pick_place_cup_side.py`) to handle
the geometry problem of a small cup in a 45 mm finger gap (only 7 mm
clearance per side on a top-down approach). The arm correctly detected
the cup and reached the staging position with the gripper rotated
horizontal — but our chosen RPY put the **servo bump pointing
downward**, and the arm dragged the servo across the ArUco sheet into
the table.

**Root cause:** the controller knows where the *flange* should be but
has no model of the gripper's collision geometry. The single tip-offset
vector tells it where the grip point is; it has no notion of the
servo's lateral protrusion. Any non-top-down RPY can put the servo into
a collision direction.

**Decision:** side approach is **deferred** until either:
1. We add a tool-collision-volume model to `MotionSettings` (a list of
   primitives in tool0 frame; before every move, transform them via the
   active RPY and check against the table plane; abort moves that
   would clip the table).
2. Or we use a real motion planner (MoveIt-style) with a full URDF.

In the meantime, **top-down (RPY=180,0,0) is the only orientation we
ship.** The servo bump stays horizontal at flange height, safely above
the table.

`pick_place_cup_side.py` is kept in the repo for reference but should
NOT be run on a configured arm without first verifying the chosen RPY
puts the servo on top — by hand-rotating the gripper or running with
hover_mm so high that any servo-down RPY can't reach the table.

---

## What the demo does step-by-step

For `pick_place_cup.py` (the top-down demo):

1. **Load profile + build motion context.** Reads
   `T_robot_world`, `table_plane`, and the ChArUco board config from
   the calibration profile. Constructs a `MotionSettings` with the
   user's CLI overrides.

2. **Warm up YOLO.** First inference on the Jetson can take 30–60 s
   for model load; we do this once before any motion to avoid a long
   blocking pause mid-trajectory.

3. **Open the iPhone (Record3D) stream.**

4. **Observe for `--detection-window-s`.** Each frame: detect the
   ChArUco board → invert to get `T_world_camera` → run YOLO segmentation
   → run the RGB-D localizer to get each detection's world XYZ. Collect
   all detections matching `--target-label` with confidence ≥
   `--min-confidence`. Bucket samples in a 5 cm world-XY grid.

5. **Pick the best bucket.** Most samples wins; ties broken by median
   confidence. Position is the median XYZ across the bucket.

6. **Apply `--xy-bias-mm`** to compensate for centroid bias.

7. **Reachability check** on both pick and place targets. Abort if
   either is out of range.

8. **Force-open the gripper (blocking).** Eliminates any race between
   `pre_grasp`'s non-blocking open and the subsequent motion.

9. **Run the FSM:**
   - `pre_grasp` → hover `--hover-mm` above the cup
   - `descend_and_grasp` → drop to 25 mm above the table, close gripper
     to `--grasp-close-value`. The AG's adaptive mechanism stalls at
     first contact with the cup body.
   - `lift` → rise `--hover-mm`
   - `place` → translate over the destination, descend to
     `--release-mm`, open gripper
   - back-off lift → rise `--hover-mm`

10. **Home** with the timing-safe sequence (1.5 s settle before, 5 s
    sleep after).

11. **Disconnect.** Leaves servos engaged so the next run starts from
    a known pose.

Every motion step logs both the **commanded** and the **actual** arm
pose (read via `get_coords_mm_deg`), and the actual gripper value, so
the run log tells you exactly what each step did. This is the
diagnostic data you'll send me when something looks wrong.

---

## Things that are known-fragile

Be aware of these. They're not bugs but they're rough edges that
require care:

- **The arm doesn't know its own tool's shape.** Any motion outside
  top-down can collide. Don't manually edit the demo to use a
  different RPY without first verifying the servo's final orientation.
- **YOLO's centroid bias varies with where the cup sits.** The
  `--xy-bias-mm 11 7` value was tuned for a cup near board center
  (world 100, 80). For cups at the far corners, the bias direction
  changes (depending on the camera angle). If you'll routinely pick
  from varied positions, custom-train YOLO on your specific cup.
- **`is_moving()` is unreliable.** We work around it with sleeps.
  Don't trust the firmware's "I'm done" signal.
- **Reach gates are flange-blind.** `is_reachable` checks the tip
  position, but the flange's robot-frame norm (which is what matters
  for joint limits) can still exceed the arm's physical envelope. The
  symptom is a silent IK rejection — the arm "barely moves" or jumps
  to a halfway pose. The mitigation is the conservative
  `--hover-mm 30`, which keeps the flange norm comfortable.
- **Servos stay engaged between runs.** If a previous run left the arm
  in an awkward pose, the next run's `power_on()` won't reset that.
  Restart the arm physically (power-cycle) for a clean start, or run
  `--release-servos` in `goto_world` first.

---

## Future work

In priority order:

1. **Tool collision model.** Define the AG body and servo bump as
   primitives in tool0 frame. Add a check in `move_to_world` that
   transforms primitive corners via the active RPY and refuses moves
   that would put any corner below the table. Unblocks side-approach
   safely, prevents another ArUco-sheet incident.

2. **Custom YOLO on the actual cup.** 30–50 annotated images, fine-tune
   `yolo26n-seg`. Should give consistent confidence (0.85+) and a more
   accurate centroid (cup-shaped objects in the COCO `cup` class look
   nothing like a red shot glass).

3. **Cup base instead of centroid.** Fit a vertical cylinder to the
   mask + depth and use the cylinder axis's intersection with the
   table plane as the cup's XY. Removes the centroid bias entirely.

4. **Cartesian-interp transit for fluid scenarios.** Current code uses
   joint-interp (`coord_mode=0`) so endpoint orientation is preserved
   but intermediate joints might tilt the cup briefly. If we ever need
   to carry actual liquid, switch transit segments to cartesian interp.

5. **Wrist camera.** A small camera on the gripper itself (which the
   AG can take) would enable a visual-servo correction at hover: take
   a top-down image right above the cup, refine the XY, then descend.
   Eliminates the centroid-bias problem for the only step that matters
   (the grasp).

---

## Files involved

| File | Role |
|---|---|
| `perception/demos/pick_place_cup.py` | The working top-down demo. |
| `perception/demos/pick_place_cup_side.py` | Side-approach demo (DEFERRED — do not run without collision model). |
| `perception/demos/goto_world.py` | Single-point arm move; used for verification and tip-offset calibration. |
| `perception/demos/yolo_world_monitor.py` | Live YOLO + ChArUco visualization, used as the ground truth for what the camera sees. |
| `perception/control/motion_primitives.py` | `MotionSettings` (tool offsets, reach limits, RPY), `MotionContext`, `move_to_world` (now RPY-aware), composed primitives (`pre_grasp`, `descend_and_grasp`, `lift`, `place`, `home`). |
| `perception/control/gripper.py` | Gripper API. `invert_polarity` lives here. |
| `perception/control/mycobot_driver.py` | Low-level arm driver. |
| `perception/calibration/profiles.py` | Calibration profile I/O. |
| `perception/calibration/charuco_board.py` | ChArUco detection. |
| `perception/localization/rgbd_localizer.py` | 2D detection → 3D world position. |
| `calibration/profiles/session_multitag.json` | Saved calibration. |
| `perception/CALIBRATION.md` | Sister doc covering how to (re)calibrate. |

---

## Quick-reference: one-liner that should just work

For a small cup placed near the board center:

```
python3 -m perception.demos.pick_place_cup --port /dev/ttyUSB0 \
    --classes "cup,bottle,wine glass,bowl,vase" \
    --target-label cup \
    --min-confidence 0.25 \
    --detection-window-s 4.0 \
    --place-mm 40 40 0 \
    --grasp-close-value 70 \
    --xy-bias-mm 11 7 \
    --hover-mm 30
```

If you change the cup or the iPhone position, the only flag likely to
need re-tuning is `--xy-bias-mm` — re-derive it by comparing the
detected position to the cup's actual position once.
