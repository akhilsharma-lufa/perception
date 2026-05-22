# Robot–World Calibration (ChArUco + Touch)

End-to-end guide for calibrating `T_robot_world` on a MyCobot 280 + iPhone 16 Pro
LiDAR (Record3D over USB to a Jetson). After calibration, the robot can drive
the tool tip to any commanded world point on the board within ~5-15 mm.

Two parallel methods are documented:

- **`touch_calibrate`** — manual touch of the pointer tip to known ChArUco
  corners. **Recommended.** Highly accurate (sub-mm world position, encoder
  precision robot position). Takes 2-5 minutes.
- **`auto_calibrate_robot`** — fully automated sweep. Faster operator time but
  relies on depth-peak detection of a 12 mm pointer, which is at the edge of
  the iPhone LiDAR's resolving power; fits typically run > 30 mm RMSE.

If you only need it to work, do **touch_calibrate**. The auto sweep is
documented for completeness and as a fallback if you can't free-drive the arm
by hand.

---

## What "calibrated" means here

The calibration produces `T_robot_world`, a 4×4 rigid transform that maps a
point in **world frame** (defined by the ChArUco board) to **robot frame**
(the MyCobot 280's base). Once saved, every downstream tool — `goto_world`,
the YOLO world monitor, pick-and-place primitives — uses this transform to
translate "the cup is at world XYZ" into joint commands.

A complete profile (`calibration/profiles/session_multitag.json`) holds:

- `robot_world_transform` — the 4×4 from calibration
- `table_plane` — board surface normal + origin, used for "hover above" math
- `charuco_board` — board geometry so the runtime monitor can detect the same
  board you calibrated with

---

## Hardware setup

### 1. Print a ChArUco board

- **Geometry used in this guide**: 11 columns × 8 rows of squares, each square
  20 mm, ArUco markers 14 mm, dictionary `DICT_4X4_50`.
- **Marker placement**: the top-left square of the board should contain a tag
  (an ArUco marker), not a black square. This is the OpenCV "legacy pattern"
  layout and is what most pre-printed boards use.
- **Print scale**: confirm with a ruler that the squares actually measure
  20 mm. Most printer drivers default to "fit to page" which scales the print
  silently. Use "actual size" / 100 % scale.
- **Mount it flat**: tape all four corners to the table; warped paper will
  hurt detection.

If your board geometry differs, you'll override with CLI flags:
`--squares-x N --squares-y M --square-mm <mm> --marker-mm <mm> --dict <NAME>`.

### 2. Pointer for the calibration

- Remove the gripper.
- Install a single thin pointer (a panel pin, the back of a black marker, a
  short bolt — anything narrow and rigid) at the center of the end face
  along the tool-Z axis.
- Measure the protrusion in mm from the end-face center to the tip.
  **Document the exact length** — this is `--tip-offset-mm` later. The
  guide uses **12 mm**.

### 3. iPhone + Record3D

- iPhone 16 Pro on a tripod, aimed down at the table such that the entire
  ChArUco board sits well within frame.
- Record3D iOS app set to **15 fps** (higher will drop frames on the Jetson's
  USB throughput).
- iOS Auto-Lock set to **Never** (Settings → Display & Brightness). Otherwise
  the screen will sleep mid-calibration and Record3D's frames degrade.
- Wired to the Jetson via the iPhone's native USB-C cable. Confirm the
  Jetson sees it with `lsusb`.

### 4. Workspace orientation

The arm needs an obstacle-free hemisphere. The MyCobot 280 has a ~28 cm
reach; monitors, cables, or shelves on the wrong side will collide. Make a
quick mental note of which side of the base is clear. The `--j1-center` flag
(see auto calibration below) lets you point the sweep at that side without
touching code.

---

## Software prerequisites

On the Jetson (everything runs on the Jetson, never the Mac):

```bash
pip install opencv-contrib-python
# (the contrib package is required; base opencv-python lacks cv2.aruco)

# Confirm:
python3 -c "import cv2; print(cv2.__version__); print(hasattr(cv2.aruco, 'CharucoBoard'))"
```

Expected: `4.10.0` (or newer) and `True`.

If you skip the contrib install, `cv2.aruco` will not be importable and every
calibration script will fail at startup with an `AttributeError`.

---

## Method 1 — Touch calibration (recommended)

You'll manually touch the pointer tip to known corners of the board, in
either free-drive (servos released) or by hand-guiding the arm. Three or more
non-collinear touches give a Kabsch fit.

### Step 1 — verify board detection

```bash
python3 -m perception.demos.touch_calibrate \
    --port /dev/ttyUSB0 \
    --squares-x 11 --squares-y 8 \
    --square-mm 20 --marker-mm 14 \
    --dict DICT_4X4_50 \
    --tip-offset-mm 12
```

The script first prints a 3 s detection trace:

```
[touch-cal] verifying ChArUco detection (3s)...
[touch-cal]   board: t=( +0.0, +0.0, +566.0) mm  corners=70  reproj=0.55 px
...
[touch-cal] OK: 8 detection(s).
```

Look for `corners` close to the maximum (`(squares_x − 1) × (squares_y − 1) =
70` for 11×8) and `reproj < 1.0 px`. If you see 0 detections, see
**Troubleshooting** below.

### Step 2 — free-drive the arm to corners

The script then releases the servos so you can move the arm by hand. Support
the arm so it does not droop under gravity.

The available target names:

- **Outer board-edge corners** (easiest — they're on the paper edge):
  - `TL` → world `(0, 0, 0)` mm
  - `TR` → world `(220, 0, 0)` mm
  - `BL` → world `(0, 160, 0)` mm
  - `BR` → world `(220, 160, 0)` mm

- **Inner chessboard corners** by `(col, row)` (0-indexed):
  - Format: `col,row` — first number is column (0 to `squares_x − 2`), second is row.
  - World position: `((col+1) × 20 mm, (row+1) × 20 mm, 0)`.
  - Example: `4,3` is the 5th column, 4th row inner corner at world `(100, 80, 0)`.
  - **Physical identification**: inner corner `(col, row)` is the
    BOTTOM-RIGHT corner of the square at grid position `(col, row)` where
    the top-left square of the board is `(0, 0)`.

### Step 3 — record at least 4 samples

Recommended set: 3 non-collinear samples gives an exact (but unverified) fit;
4+ samples give a real RMSE.

```
[touch-cal] > free
                              # move arm so tip touches BR outer corner
[touch-cal] > BR              # records sample, auto-locks servos
[touch-cal] > free
                              # move arm so tip touches TL outer corner
[touch-cal] > TL
[touch-cal] > free
                              # move arm so tip touches center inner corner (4,3)
[touch-cal] > 4,3
[touch-cal] > free
                              # move arm so tip touches one more, e.g. (5,2)
[touch-cal] > 5,2
[touch-cal] > fit
```

The `fit` command runs Kabsch. Look for:

```
[touch-cal] Kabsch RMSE = 2.54 mm  (4 samples)
[touch-cal]    0          BR   residual =  1.10 mm
[touch-cal]    1          TL   residual =  2.40 mm
[touch-cal]    2 inner(4,3)    residual =  3.80 mm
[touch-cal]    3 inner(5,2)    residual =  2.50 mm
[touch-cal] T_robot_world =
[[-1.00  0.02  0.00  0.24]
 [ 0.02  1.00  0.08  0.11]
 [ 0.00  0.08 -1.00 -0.02]
 [ 0.00  0.00  0.00  1.00]]
```

- **RMSE < 5 mm** → excellent, save it.
- **RMSE 5-10 mm** → acceptable, but one of your touches was sloppy.
- **RMSE > 15 mm** → check which sample has the highest residual, drop it
  with `drop N`, and refit. Or re-touch that point.

### Step 4 — save

```
[touch-cal] > save
[touch-cal] > quit
```

The profile at `calibration/profiles/session_multitag.json` now has
`robot_world_transform` populated.

### Step 5 — APPLY THE TWO REQUIRED CORRECTIONS

There are two known traps in this stack that are silently applied by the
current code, but you should understand them so you can debug if they ever
get reverted.

#### Correction 1 — table normal direction

**Problem**: OpenCV's `CharucoBoard` convention places the board's +Z axis
AWAY from the markers — so for a board lying flat on the table with markers
facing up at the camera, the board's +Z points DOWN INTO the table. If you
save `table_plane.normal_world = (0, 0, +1)`, the `motion_primitives.project_above_table`
function will then "lift" hover targets in the +Z direction, which is DOWN
into the table.

**Symptom**: `goto_world --hover-mm 30` makes the tip drag along the paper
instead of hovering 30 mm above it. You see the tip plunge through the table
during the cartesian move.

**Fix**: save `table_plane.normal_world = (0, 0, -1)`. The newer versions of
`touch_calibrate.py` and `auto_calibrate_robot.py` do this automatically.
If you have an older profile that was saved with the wrong sign, repair it
with this one-liner:

```bash
python3 -c "
from perception.calibration.profiles import CalibrationProfileIO
import numpy as np
p = CalibrationProfileIO.load('calibration/profiles/session_multitag.json')
p.set_table_plane(
    normal_world=np.array([0.0, 0.0, -1.0]),
    origin_world=np.array([0.0, 0.0, 0.0]),
    inlier_ratio=1.0,
    mean_abs_residual_m=0.0,
)
CalibrationProfileIO.save(p, 'calibration/profiles/session_multitag.json')
print('OK: table_normal_world flipped to (0, 0, -1)')
"
```

After this fix, `--world-mm 110 80 0 --hover-mm 30` produces:

```
[goto_world] target world (mm): (+110.0, +80.0, -30.0)   ← Z is NEGATIVE for "above"
```

That's correct. World +Z is INTO the table by board convention, so "above"
is −Z. The script handles the sign transparently when you use `--hover-mm`.

#### Correction 2 — tip offset compensation at runtime

**Problem**: `touch_calibrate.py` records the **tip** position (flange minus
the pointer length) at each touch. So `T_robot_world` learns to map
`world → tip-in-robot`. But the MyCobot's IK controller takes a FLANGE pose,
not a tip pose. If you send the tip-in-robot value as the flange target, the
flange goes where the tip should go, and the actual tip ends up
`tip_offset_z_m` BELOW the request.

**Symptom**: with the correct table normal but no tip-offset compensation,
`--hover-mm 0` makes the tip dive ~12 mm into the paper.

**Fix**: in `motion_primitives.move_to_world`, after computing the tip
target in robot frame, add `(0, 0, tip_offset_z_m)` to get the flange
target — the flange sits `tip_offset_z_m` ABOVE the tip in robot Z when the
gripper points straight down (RPY=(180,0,0)).

The current code in `perception/control/motion_primitives.py` does this:

```python
p_robot_tip = world_to_robot(p_world_m, ctx.t_robot_world)
p_robot_flange = p_robot_tip + np.array(
    [0.0, 0.0, float(ctx.settings.tip_offset_z_m)], dtype=np.float64
)
# ... send p_robot_flange as the flange target
```

`MotionSettings.tip_offset_z_m` defaults to `0.012` (12 mm). When you change
the tool (e.g., re-attach the gripper), update this value in the
`MotionSettings` you pass into the demo, or in the underlying default.

### Step 6 — verify with goto_world

```bash
# Should hover ~3 cm above the center of the board
python3 -m perception.demos.goto_world \
    --world-mm 110 80 0 --hover-mm 30 --speed 15 \
    --max-reach-mm 400 --port /dev/ttyUSB0

# Should LIGHTLY touch the center (no hover)
python3 -m perception.demos.goto_world \
    --world-mm 110 80 0 --hover-mm 0 --speed 15 \
    --max-reach-mm 400 --port /dev/ttyUSB0

# Should hover ~3 cm above the BR outer corner
python3 -m perception.demos.goto_world \
    --world-mm 220 160 0 --hover-mm 30 --speed 15 \
    --max-reach-mm 400 --port /dev/ttyUSB0
```

What good looks like:

- `target world (mm)` shows a NEGATIVE Z when `--hover-mm > 0` — that's
  the correction working (recall world +Z is into the table).
- `arm tip moved` is several hundred mm (not 0, which would mean IK failed).
- `error vs commanded target` is < 15 mm.
- Visually: at `--hover-mm 0`, the tip touches the requested corner cleanly.
  At `--hover-mm 30`, the tip is a clear finger-width above the paper.

If `arm tip moved` is < 5 mm, IK rejected the target. Try:
- Bumping `--hover-mm` (sometimes higher hover gives the wrist more
  joint-space freedom).
- Passing `--rpy 180 0 -90` or similar to try a different wrist orientation.
- Picking a different `--world-mm` — the very corners of the board can be
  past the arm's reach envelope.

---

## Method 2 — Automated sweep (fallback)

This method moves the arm through ~22 predetermined joint configurations,
samples the iPhone depth at each, finds the highest point above the board
plane (assumed to be the pointer tip), and fits Kabsch on the resulting
`(tip_world, tip_robot)` pairs.

### Why this is worse than touch

- A 12 mm pointer at 56 cm from the iPhone covers ~3 depth pixels. Depth at
  those pixels gets averaged with the background → noisy, biased Z.
- The "highest point above the board" assumption is brittle: static objects
  in the frame (robot base, monitor edges) often outrank the actual tip.
- Some joint configurations put the tip outside the iPhone's FOV, so the
  detector locks onto a fixed scene feature for every waypoint.

Real-world result: 80-150 mm RMSE typically. The touch flow gives ~3 mm.

If you still need to run it (e.g., you cannot release the servos to free
drive, or the arm tool is too short to manually touch corners precisely),
the steps are:

### Step A — dry-run to verify board detection

```bash
python3 -m perception.demos.auto_calibrate_robot \
    --dry-run \
    --port /dev/ttyUSB0 \
    --squares-x 11 --squares-y 8 \
    --square-mm 20 --marker-mm 14 \
    --dict DICT_4X4_50
```

### Step B — safety probe (one waypoint, low risk)

Identify which side of the base is clear of obstacles by running a SINGLE
waypoint and watching the arm rotate:

```bash
python3 -m perception.demos.auto_calibrate_robot \
    --port /dev/ttyUSB0 \
    --max-waypoints 1 \
    --j1-center 90 \
    --squares-x 11 --squares-y 8 \
    --square-mm 20 --marker-mm 14 \
    --dict DICT_4X4_50
```

If the arm rotates INTO your obstacles, rerun with `--j1-center -90` or
`--j1-center 180` until you find the safe side. The arm homes first, then
moves to a single waypoint at `[j1_center, -50, -40, 0, -10, 0]`, then
exits.

### Step C — full sweep

Once you've confirmed the safe side, drop `--max-waypoints` and let the
22-waypoint sweep run (~3 minutes):

```bash
python3 -m perception.demos.auto_calibrate_robot \
    --port /dev/ttyUSB0 \
    --j1-center 90 \
    --squares-x 11 --squares-y 8 \
    --square-mm 20 --marker-mm 14 \
    --dict DICT_4X4_50
```

Add `--j1-center-2 -90` (if your right side is also clear) to sweep both
hemispheres — doubles the wall time but doubles the geometric constraint
on the fit.

The script captures a reference depth frame at home pose, then at each
waypoint masks out pixels that did NOT change vs. the reference. This is
how it filters static scene objects from the depth-peak search. It helps
but does not fully solve the fundamental noise/resolution problem.

The same two corrections (Correction 1 and Correction 2 above) apply.

---

## Troubleshooting

### "No board detected" / 0 samples in pre-flight

- Verify the print scale with a ruler. If your squares are 19 mm instead of
  20, detection might still work but the world coordinates of every corner
  shift proportionally, breaking the fit.
- Verify the dictionary. If you generated the board with a non-default
  dictionary (e.g., `DICT_5X5_100`), pass `--dict DICT_5X5_100`.
- Verify the rows × columns. If your board has 8 rows × 11 columns, that
  means `--squares-x 11 --squares-y 8`. The X count is the LONGER side
  (running along the long edge of the paper).
- Make sure the iPhone screen is awake. iOS auto-lock kills Record3D's
  frame quality silently.
- Check lighting. Direct overhead light reflects off white paper and
  blinds the marker corners. Diffuse / side lighting is better.

### "Markers detected but 0 ChArUco corners"

This is the **legacy pattern** issue. OpenCV 4.7+ changed where it places
ArUco IDs on a ChArUco board. The new layout has the top-left square BLACK;
the legacy (≤ 4.6) layout has the top-left square as a marker. Most
pre-printed boards use the legacy layout.

Symptom: `cv2.aruco.detectMarkers` returns ~40 markers but
`CharucoDetector.detectBoard` returns 0 corners.

Fix: set `legacy_pattern=True` on `CharucoBoardConfig`. The newer demo
scripts default to True; the runtime monitor reads it from the saved
profile. If you ever generate a board with OpenCV 4.7+ and the new layout,
pass `--no-legacy-pattern`.

Debug tool: `python3 -m perception.demos.debug_charuco --frames 5` prints
markers found, corners interpolated, solvePnP result, and saves an
annotated frame to `charuco_debug.png`.

### "arm tip moved 0.5 mm" / IK rejected

The pymycobot IK could not find joint angles that satisfy your XYZ target
WITH the requested RPY=(180,0,0). Either:
- The target is past the reach envelope (cartesian distance > 27 cm from
  base).
- The wrist needs a near-singular joint configuration that pymycobot
  refuses.
- The target Z combined with this XY puts the wrist past a joint limit.

Workarounds:
- Use a different `--hover-mm` (sometimes higher Z helps — paradoxically,
  the IK rejects some specific cartesian values but accepts the same
  XY at a different Z).
- Pass `--rpy 180 0 -90` to try a different wrist roll.
- Try a less aggressive target (closer to the center of the workspace).

### Tip touches table at `--hover-mm 0` but is offset by ~10-15 mm in XY

That's MyCobot 280's inherent cartesian repeatability. The arm spec sheet
quotes ~5 mm; in practice with the AG mount and standard firmware, ±10 mm
in XY is realistic. There is no software fix — it's the floor of what the
hardware delivers.

For pick-and-place of small cups (4 cm diameter), 10-15 mm position error
is absorbed by the gripper's open width (20-45 mm). For tasks needing
better precision, you'd need a more expensive arm.

### "Tip touches the table when I asked for hover=30"

You either skipped one of the two corrections above, or you re-ran an
older `auto_calibrate_robot` that re-wrote the profile with the wrong
table normal. Verify:

```bash
python3 -c "
from perception.calibration.profiles import CalibrationProfileIO
p = CalibrationProfileIO.load('calibration/profiles/session_multitag.json')
print('table_plane.normal_world =', p.table_plane.normal_world)
"
```

If it prints `[0.0, 0.0, 1.0]` (positive Z), apply Correction 1's
one-liner from Step 5 above.

---

## File reference

Files involved in this calibration flow:

| File | Role |
|---|---|
| `perception/demos/touch_calibrate.py` | Interactive manual touch calibration (recommended) |
| `perception/demos/auto_calibrate_robot.py` | Automated waypoint-sweep calibration |
| `perception/demos/debug_charuco.py` | Single-frame ChArUco detection debugger |
| `perception/demos/goto_world.py` | Verification: drive the tool tip to a world point |
| `perception/calibration/charuco_board.py` | ChArUco detection + pose solver |
| `perception/calibration/tip_detector.py` | Depth-peak pointer detector (auto path) |
| `perception/calibration/auto_robot_calibrator.py` | Waypoint generator + Kabsch fit |
| `perception/calibration/robot_calibrator.py` | Kabsch math + tip-offset helper |
| `perception/calibration/profiles.py` | Profile dataclass + JSON I/O |
| `perception/control/motion_primitives.py` | `move_to_world` (where Correction 2 lives) |
| `calibration/profiles/session_multitag.json` | Saved profile output |

## Changing the tool

When you change the tool (pointer → gripper, or to a different gripper),
update `MotionSettings.tip_offset_z_m` to the new flange-to-tip distance in
meters. The XY calibration carries over because it depends only on
`T_robot_world`, which is tool-independent. You do NOT need to redo the
ChArUco touch calibration — just the tip offset constant.

For the AG gripper, the flange-to-tip distance is approximately 110 mm
(`tip_offset_z_m = 0.110`). Verify with calipers or a simple `goto_world`
test at hover=50.

---

## Summary — minimal commands once everything is set up

```bash
# 1) install once
pip install opencv-contrib-python

# 2) calibrate (interactive)
python3 -m perception.demos.touch_calibrate \
    --port /dev/ttyUSB0 \
    --squares-x 11 --squares-y 8 --square-mm 20 --marker-mm 14 \
    --tip-offset-mm 12

# inside the prompt:
#   free, BR, free, TL, free, 4,3, free, 5,2, fit, save, quit

# 3) verify
python3 -m perception.demos.goto_world \
    --world-mm 110 80 0 --hover-mm 30 --speed 15 \
    --max-reach-mm 400 --port /dev/ttyUSB0
```

If the tip hovers cleanly above the center of the board, you're done.
