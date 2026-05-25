# Touch-Calibration Command Reference

Quick reference for the interactive prompt you get when you run:

```bash
python -m perception.cup_locator.calibrate \
    --port /dev/ttyUSB0 \
    --profile calibration/profiles/session_multitag.json \
    --squares-x 10 --squares-y 7 --square-mm 20 --marker-mm 15 \
    --dict DICT_4X4_50 \
    --tip-offset-mm 12
```

The prompt is `[touch-cal] >`. Type a command, press Enter. Commands are case-insensitive.

---

## Recording touches

These commands record one `(world_position, tip_in_robot)` correspondence and append it to the in-memory sample list.

| Command | What it touches | World position (m) |
|---|---|---|
| `TL` | Outer corner at the **top-left** of the printed grid | `(0, 0, 0)` |
| `TR` | Outer corner at the **top-right** | `(sx, 0, 0)` |
| `BL` | Outer corner at the **bottom-left** | `(0, sy, 0)` |
| `BR` | Outer corner at the **bottom-right** | `(sx, sy, 0)` |
| `col,row` | **Inner** chessboard intersection (X-shape where 4 squares meet) | `((col+1)·s, (row+1)·s, 0)` |

Where `s = square_length_m`, `sx = squares_x · s`, `sy = squares_y · s`.

For inner corners, `col ∈ [0 .. squares_x − 2]` and `row ∈ [0 .. squares_y − 2]` (0-indexed). On a 10×7 board: `col ∈ [0..8]`, `row ∈ [0..5]`.

You can also type `col row` with a space instead of a comma — both parse the same.

**Side effect:** if the arm is currently in free-drive mode, the script auto-locks the servos before reading the pose (so the arm doesn't drift under gravity), sleeps 100 ms for the pose to settle, then reads `get_coords_mm_deg`. It will then *stay* locked until you type `free` again. The prompt prints `(auto-locked to capture stable pose)` when this happens.

---

## Moving the arm

| Command | Effect |
|---|---|
| `free` | Release all servos. The arm goes limp — **support it with your free hand** so it doesn't drop under gravity. Use this between touches to reposition the gripper by hand. |
| `lock` | Re-engage the servos at the arm's current pose. The arm freezes where you left it. Useful if you want to inspect a touch before recording it, or if you're stepping away from the rig. |

The script starts in `free` mode unless you pass `--start-locked`.

---

## Inspecting state

| Command | Effect |
|---|---|
| `show` | Print the current `get_angles_deg()`, `get_coords_mm_deg()`, the tip-in-robot position (flange minus `tip_offset_z_m` along Z), and whether the servos are locked. Doesn't record anything. |
| `list` | Print every sample recorded so far, with its index, target name, world position (mm), and tip-in-robot position (mm). Use this to check progress before `fit`. |

---

## Editing samples

| Command | Effect |
|---|---|
| `drop N` | Remove sample number `N` (the index `list` prints next to each row). Use this when `fit` shows one sample with a residual much larger than the others — that touch was probably mis-pointed; drop it and re-touch the same target. |

You can drop and re-record in any order. The sample list is just a Python list under the hood.

---

## Solving and saving

| Command | Effect |
|---|---|
| `fit` | Run the Kabsch SVD alignment on the current sample set. Prints the RMSE in mm, the per-point residual for every sample, and the 4×4 `T_robot_world` matrix. Requires **≥ 3 non-collinear** samples — fewer and it refuses with a message. Does **not** write anything to disk. |
| `save` | Write the most recently fitted `T_robot_world` to the profile JSON specified by `--profile`. If the profile file already exists, it loads, mutates, and overwrites it (preserving any non-related fields). If it doesn't exist, a fresh `v3-charuco` profile is created. Requires that you ran `fit` since the last edit — otherwise it tells you `nothing to save; run 'fit' first.` |

What `save` actually persists into the JSON:

- `robot_world_transform`: the fitted 4×4 matrix.
- `charuco_board`: the board spec you passed on the CLI (`squares_x/y`, `square_length_m`, `marker_length_m`, `dictionary_name`, `legacy_pattern`).
- `table_plane`: `{normal = (0, 0, -1), origin = (0, 0, 0), inlier_ratio = 1.0, mean_abs_residual_m = 0.0}` (a synthetic plane perpendicular to the board, with the sign chosen so "above the table" lifts toward the camera).
- `metrics`: `{robot_world_n_samples, tip_offset_z_m, calibration_method: 1.0}`.
- `created_at_utc`: a fresh ISO-8601 timestamp.

You can `fit` and `save` repeatedly — each `save` writes the latest fit.

---

## Exiting

| Command | Effect |
|---|---|
| `quit` | Exit the prompt loop. Aliases: `q`, `exit`. Also triggered by EOF (Ctrl-D). |

On exit, the script:

1. Re-engages the servos at whatever pose the arm is in (so it doesn't drop under gravity once the controller disconnects).
2. Disconnects the MyCobot.
3. Disconnects the Record3D camera.

If you `quit` without `save`-ing, your sample list is lost. You'll get no warning — be deliberate.

---

## Common error messages

| Message | What it means | Fix |
|---|---|---|
| `unknown command 'X'` | What you typed doesn't match any command or target spec. | Re-read the prompt help line; check for typos. `col,row` must be two integers. |
| `col N out of range [0, K]` | Inner-corner index out of bounds for the configured board geometry. | Re-check your `--squares-x/y` flags vs. the printed board. |
| `need at least 3 non-collinear samples to fit Kabsch; have N` | You ran `fit` with fewer than 3 samples. | Record more touches. |
| `nothing to save; run 'fit' first.` | You ran `save` before any successful `fit`. | Run `fit`, then `save`. |
| `[touch-cal] FATAL: no board detections.` | The verifier saw zero ChArUco detections in the warmup period. | Check `--squares-x/y`, `--square-mm`, `--dict`, and `--legacy-pattern` against the printed board. Confirm the board is fully in view and well-lit. |
| `warn: release_all_servos failed` | The arm wouldn't release. Usually a transient comms hiccup. | Try `free` again, or check the USB cable. |

---

## A complete session at a glance

```text
$ python -m perception.cup_locator.calibrate --port /dev/ttyUSB0 \
    --profile calibration/profiles/session_multitag.json \
    --squares-x 10 --squares-y 7 --square-mm 20 --marker-mm 15 \
    --dict DICT_4X4_50 --tip-offset-mm 12

[touch-cal] verifying ChArUco detection (3s)...
[touch-cal]   board: t=(+12.4, -45.7, +412.0) mm  corners=29  reproj=0.68 px
[touch-cal] OK: 11 detection(s).
[touch-cal] connecting to MyCobot on /dev/ttyUSB0...

[touch-cal] AVAILABLE TARGETS
  Outer corners (paper-edge):
    TL  world = (   0.0,    0.0,    0.0) mm
    TR  world = ( 200.0,    0.0,    0.0) mm
    BL  world = (   0.0,  140.0,    0.0) mm
    BR  world = ( 200.0,  140.0,    0.0) mm
  Inner corners: col in [0..8], row in [0..5], world = ((col+1)*20.0, (row+1)*20.0, 0) mm

[touch-cal] releasing servos for free-drive. ...

[touch-cal] > 2,2          # touch inner (2,2) → world (60, 60, 0) mm
  (auto-locked to capture stable pose)
  recorded inner(2,2): world=(+60.0, +60.0, +0.0) mm  tip_robot=(...) mm. (1 sample(s) total)
[touch-cal] > free
[touch-cal] > 7,2          # touch inner (7,2) → world (160, 60, 0) mm
  recorded inner(7,2): ... (2 sample(s) total)
[touch-cal] > free
[touch-cal] > 2,4          # touch inner (2,4) → world (60, 100, 0) mm
  recorded inner(2,4): ... (3 sample(s) total)
[touch-cal] > free
[touch-cal] > 7,4
  recorded inner(7,4): ... (4 sample(s) total)
[touch-cal] > fit
[touch-cal] Kabsch RMSE = 2.31 mm  (4 samples)
[touch-cal]      0    inner(2,2)  residual =  1.85 mm
[touch-cal]      1    inner(7,2)  residual =  2.71 mm
[touch-cal]      2    inner(2,4)  residual =  2.04 mm
[touch-cal]      3    inner(7,4)  residual =  2.59 mm
[touch-cal] T_robot_world =
 [[ ... 4x4 ... ]]
[touch-cal] > save
  loaded existing profile: calibration/profiles/session_multitag.json
  saved -> calibration/profiles/session_multitag.json
[touch-cal] > quit
[touch-cal] locking servos before exit (so the arm doesn't drop).
```

---

## Cheat-sheet

```
free            release servos (move arm by hand)
lock            re-engage servos at current pose
TL/TR/BL/BR     record an outer-corner touch
col,row         record an inner-corner touch (0-indexed)
show            print current pose + board detection
list            list recorded samples
drop N          remove sample N from the list
fit             run Kabsch, print RMSE + residuals
save            write the fit to the profile JSON
quit            exit (aliases: q, exit, Ctrl-D)
```
