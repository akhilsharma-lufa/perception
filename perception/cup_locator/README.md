# `perception.cup_locator` — Usage Guide

A small subpackage that hides the perception stack (iPhone camera → ChArUco anchor → YOLO segmentation → RGB-D localization → world-frame tracker with smoothing) behind a single blocking call: detect one cup, return its coordinates in the **MyCobot 280 base frame (mm)**.

You don't need to know anything about the perception layers below. You construct `CupLocator`, call `.locate()`, get a `CupPose`, and pass its `position_robot_mm` into your path planner.

---

## Contents

1. [Mental model](#1-mental-model)
2. [What you received](#2-what-you-received)
3. [Installation](#3-installation)
4. [The calibration profile — what it is and when it's valid](#4-the-calibration-profile)
5. [Loading the profile in your code](#5-loading-the-profile-in-your-code)
6. [`CupLocator` — full constructor reference](#6-cuplocator--full-constructor-reference)
7. [Methods](#7-methods)
8. [`CupPose` — what `locate()` returns](#8-cuppose--what-locate-returns)
9. [Coordinate conventions](#9-coordinate-conventions)
10. [Usage patterns](#10-usage-patterns)
11. [Performance & timing](#11-performance--timing)
12. [Tuning for your environment](#12-tuning-for-your-environment)
13. [Troubleshooting](#13-troubleshooting)
14. [Recalibrating](#14-recalibrating)
15. [Limitations & non-goals](#15-limitations--non-goals)
16. [Quick reference card](#16-quick-reference-card)

---

## 1. Mental model

There are three coordinate frames involved. You only need to read from one (robot), but knowing all three helps when you debug:

```
   world frame                 camera frame              robot base frame
   (ChArUco board origin)      (live, per-frame)         (MyCobot 280 base)
            ↑                         ↑                          ↑
            └── live ChArUco          └── live ChArUco            └── fixed
                pose estimation           pose estimation             T_robot_world
                                                                      from calibration
```

What `CupLocator.locate()` does, end to end:

1. Pulls one Record3D frame (RGB + depth + intrinsics).
2. Runs `detect_board_pose` on the RGB to find the ChArUco board → gives `T_world_camera`.
3. Runs YOLO segmentation on the RGB → gives a list of object masks (filtered to your target label).
4. Unprojects the depth pixels under each mask to 3-D, transforms into world frame via `T_world_camera`, takes a robust percentile centroid → world-frame XYZ in meters.
5. Updates the `WorldTracker` with all detected objects. Tracker smooths position (EMA), height (median over a window), yaw (circular EMA), and quality (EMA).
6. Once the same tracked cup has been observed for ≥ `settle_frames` consecutive frames with smoothed quality ≥ `min_quality`, it transforms the world-frame centroid into the robot base frame via the calibrated `T_robot_world` and returns it.

If the cup hasn't settled within `timeout_s`, returns `None`.

You never call any of these layers directly — `CupLocator` orchestrates them.

---

## 2. What you received

Three things, all required:

| Item | What it is | Where to put it |
|---|---|---|
| `perception/` directory | Python package containing the locator and its dependencies. | On your `PYTHONPATH` (e.g. project root). |
| `calibration/profiles/session_multitag.json` | The output of touch-calibration. Encodes `T_robot_world`, the ChArUco board geometry, and the table plane. | Anywhere; you pass the path to `CupLocator(...)`. |
| `yolo26n-seg.pt` | The YOLO segmentation model file (~ a few MB). | In your working directory, OR set `model_path=...` on `CupLocator`. |

You do **not** need (and should not call) `touch_calibrate.py`, `goto_world.py`, or anything in `perception/demos/`. Those are calibration / verification tools on the sender's side. The contract you have is the profile JSON + the API documented here.

---

## 3. Installation

```bash
pip install numpy opencv-contrib-python record3d ultralytics torch pymycobot
```

- `opencv-contrib-python` (not plain `opencv-python`) — ChArUco lives in `cv2.aruco`, only in the contrib build.
- `record3d` — iPhone camera driver. The locator opens a Record3D session when `locate()` is first called.
- `ultralytics` + `torch` — YOLO. On first run, ultralytics may try to validate the `.pt` file; pre-place the model file in your working directory to avoid downloads.
- `pymycobot` — transitively imported by `perception.control.motion_primitives` (for the `world_to_robot` helper). Even if you don't use the MyCobot driver yourself, the package must be installed for the import chain to succeed.

Python 3.10+ recommended (the dataclasses use `frozen=True` and `tuple[float, float, float]` annotations).

---

## 4. The calibration profile

The profile JSON is the output of the touch-calibration session that was done before handoff. It contains:

| Field | What it encodes | Used when |
|---|---|---|
| `robot_world_transform` (4×4) | The rigid transform that maps a world-frame point (meters) into the MyCobot's base frame (meters). | Every `locate()` call, to convert world XYZ → robot mm. |
| `charuco_board` | Board geometry (squares_x, squares_y, square_length_m, marker_length_m, dictionary_name, legacy_pattern). | Every frame, by the live ChArUco anchor. |
| `table_plane` | Synthetic plane: `normal = (0, 0, −1)`, `origin = (0, 0, 0)`. The board surface, with +Z into the table. | The RGB-D localizer uses it to compute cup height above the table. |
| `metrics` | Diagnostic only (RMSE, sample count, tip offset used). | Not used at runtime. |

**The profile is valid as long as the MyCobot base and the ChArUco board don't move relative to each other.** The camera can be repositioned freely — the live ChArUco anchor handles per-frame camera pose.

If either the robot or the board moves, the profile is stale; see [§14 Recalibrating](#14-recalibrating).

---

## 5. Loading the profile in your code

```python
from perception.cup_locator import CupLocator

loc = CupLocator("calibration/profiles/session_multitag.json")
```

What happens at construction:

- The JSON is read and validated. Two hard requirements: a `robot_world_transform` and a `charuco_board` block must both be present. If either is missing, `__init__` raises `ValueError` with a descriptive message — the profile wasn't fully written.
- `T_robot_world` (4×4 numpy matrix) is cached.
- A `CharucoBoardConfig` is built from the profile's `charuco_board` section. **You don't pass board dimensions on the CLI or in the constructor** — they come from the profile.
- The YOLO detector, RGB-D localizer, and world tracker are instantiated. YOLO weights are *not* loaded yet (lazy).
- The Record3D camera is *not* opened yet (lazy).

This means construction is cheap and side-effect-free. The camera opens on the first `locate()` call.

Always use the context manager so the camera is released cleanly:

```python
with CupLocator("...") as loc:
    pose = loc.locate(timeout_s=2.0)
```

---

## 6. `CupLocator` — full constructor reference

```python
CupLocator(
    profile_path: str,
    *,
    target_label: str = "cup",
    min_confidence: float = 0.15,
    settle_frames: int = 8,
    min_quality: float = 0.20,
    anchor_max_reproj_px: float = 1.5,
    model_path: str | None = None,
)
```

| Argument | Default | What it controls | When to change |
|---|---|---|---|
| `profile_path` | — | Filesystem path to the calibration profile JSON. | Always — pass your profile path. |
| `target_label` | `"cup"` | YOLO class label to detect. Detections of other classes are dropped. | If your cup is classified differently by YOLO (`"wine glass"`, `"bowl"`, `"vase"`, `"bottle"`) — verify by running `cup_locator_demo` first. |
| `min_confidence` | `0.15` | YOLO confidence floor. Detections below this are ignored before they ever reach the tracker. Lowered from the default 0.30 to favour reliability over precision. | Raise (e.g. 0.30) if you're getting false positives. Lower (e.g. 0.05) if reliable cups are being dropped. |
| `settle_frames` | `8` | How many consecutive tracker hits the same cup must accumulate before `locate()` returns it. Higher → more stable but slower. | Raise for noisier scenes. Lower (2–3) for fast, single-shot demos where any detection will do. |
| `min_quality` | `0.20` | Minimum smoothed quality (0–1) for a track to count toward the settle gate. Quality is a composite of mask depth-support ratio and per-pixel depth confidence. | Raise if low-quality detections are being returned. |
| `anchor_max_reproj_px` | `1.5` | Reject ChArUco detections whose reprojection error exceeds this. A loose anchor pollutes downstream XYZ. | Raise if your camera is noisy (1.5 px is conservative; 2.5 px is usually still fine). |
| `model_path` | `None` | If set, overrides the default YOLO model `"yolo26n-seg.pt"`. | If you have a custom-trained segmentation model. |

All keyword arguments are independent — defaults are tuned to a typical desk setup with a 10×7 ChArUco board and a 5 cm cup.

---

## 7. Methods

### `locate(timeout_s=2.0, settle_frames=None, *, device_index=0) -> CupPose | None`

Blocks until one stable cup pose is available or returns `None` on timeout.

- `timeout_s` — wall-clock budget. The loop pulls camera frames (each frame ≈ 30–60 ms), so a 2 s timeout allows 30+ frames of accumulation.
- `settle_frames` — overrides the constructor's value for this call only. Useful for a "fast first read, careful subsequent reads" pattern.
- `device_index` — Record3D device to open. Only matters on the **first** call; subsequent calls reuse the camera.

If multiple matching tracks satisfy the settle gate (e.g. two cups on the table), the one with the highest smoothed quality wins.

The camera is opened **lazily** on first call. Subsequent calls reuse the same Record3D session and the world tracker's accumulated smoothing state — so second-onward calls are faster and more stable than the first.

### `recalibrate(**touch_cal_kwargs) -> None`

Runs the full interactive touch-calibration flow against this instance's profile path, then reloads. Forwards keyword arguments as `--flag value` CLI arguments (underscores become dashes, booleans become bare `--flag` / `--no-flag`).

Requires the MyCobot to be connected. Useful only if the robot or board has moved mid-session.

For the touch-cal prompt commands themselves, see `perception/cup_locator/CALIBRATION.md`.

### `close() -> None`

Releases the Record3D camera. Idempotent. Also called via the context-manager `__exit__`, so prefer:

```python
with CupLocator("...") as loc:
    ...
```

over manual `close()`.

### Context manager

`CupLocator` supports `with`:

```python
with CupLocator(profile_path) as loc:
    pose1 = loc.locate()
    pose2 = loc.locate()   # reuses camera + smoothing state
# camera released here
```

This is the recommended pattern. The instance can be used across many `locate()` calls; the camera + smoother stay alive between them.

---

## 8. `CupPose` — what `locate()` returns

```python
@dataclass(frozen=True)
class CupPose:
    position_robot_mm: tuple[float, float, float]
    position_world_m:  tuple[float, float, float]
    height_m:          float | None
    quality:           float
    yaw_hint_rad:      float | None
    label:             str
    track_id:          str
```

### Fields, in detail

| Field | Type | Meaning | Typical use |
|---|---|---|---|
| `position_robot_mm` | `(x_mm, y_mm, z_mm)` | **Primary output.** Cup centroid in MyCobot 280 base frame, millimeters. Same frame as `mc.get_coords()` and `mc.send_coords()`. | Plug into `send_coords([x, y, z, rx, ry, rz], speed, mode)` along with the RPY you choose for the grasp. |
| `position_world_m` | `(x_m, y_m, z_m)` | Cup centroid in the ChArUco board frame, meters. Origin at the marker corner, +X along columns, +Y along rows, +Z **into** the table. | Debugging; sanity-checking where the cup is on the board ("is X about 100 mm? that's column 5 mid-board"). |
| `height_m` | `float` or `None` | Estimated cup height above the table, meters. Smoothed by a rolling median over 15 frames. `None` if the localizer couldn't estimate it (mask too small, depth too sparse). | Use to gate "is this object tall enough to grasp from above?" or "is this short enough that my gripper body would hit the table?". Not surgically accurate — expect ±1 cm on a small cup. |
| `quality` | `float` (0..1) | Smoothed track quality. Composite of mask depth-support ratio and mean depth confidence. | Below 0.3 = noisy; consider rejecting. Above 0.5 = solid detection. |
| `yaw_hint_rad` | `float` or `None` | Suggested gripper yaw (rotation about world Z), radians, from a PCA of the cup's mask points. `None` for cups (symmetric). | If you ever extend the locator to handle elongated objects (bottles, spoons), this becomes useful. For cups, ignore. |
| `label` | `str` | The YOLO class that triggered the detection. | Sanity check (`assert pose.label == "cup"`); also useful if you set `target_label` to something other than `"cup"`. |
| `track_id` | `str` | Persistent identifier across calls, e.g. `"cup-1"`. Format: `"{label}-{tracker_int_id}"`. | Detecting "is this the same cup as last time?". If the id changes between calls, the tracker dropped the old track (e.g. cup moved out of view for >12 frames). |

### Frame conventions visualised

```
World (board) frame:
    Origin: ChArUco TL marker corner
    +X: along the 11-square edge (long edge)
    +Y: along the 8-square edge (short edge)
    +Z: INTO the table (so the cup's centroid has negative Z)

Robot base frame:
    Origin, +X, +Y, +Z: whatever pymycobot's get_coords() reports for your arm.
    Same as send_coords()' XYZ input.
```

`position_robot_mm` is computed inside the locator as:

```python
p_world_m = numpy.array(cup_centroid_in_world_m)
p_robot_m = world_to_robot(p_world_m, T_robot_world)          # 4×4 from profile
position_robot_mm = tuple(c * 1000.0 for c in p_robot_m)
```

So the same `T_robot_world` that the sender backed out via touch-calibration is what drives every coordinate you see.

---

## 9. Coordinate conventions

| Frame | Origin | +X | +Y | +Z | Units |
|---|---|---|---|---|---|
| World | ChArUco TL marker corner | Along long edge of grid (11 squares) | Along short edge (8 squares) | Into the table | meters |
| Robot base | MyCobot 280 base zero (pymycobot's convention) | (your robot's +X) | (your robot's +Y) | (your robot's +Z) | millimeters in API |

To pair `position_robot_mm` with a gripper pose, the canonical "top-down grasp" RPY for the MyCobot AG is `(180, 0, 0)`:

```python
x_mm, y_mm, z_mm = pose.position_robot_mm
mc.send_coords([x_mm, y_mm, z_mm + HOVER_MM, 180.0, 0.0, 0.0], speed=25, mode=0)
```

Side approaches need different RPYs; see the diff against `perception/demos/pick_place_cup_side.py` for examples (with the known caveat that the demo's `_APPROACH_RPYS` table doesn't currently roll the gripper to keep the servo housing on top — you'll need to add that yourself, or use top-down).

---

## 10. Usage patterns

### One-shot detection then exit

```python
from perception.cup_locator import CupLocator

with CupLocator("calibration/profiles/session_multitag.json") as loc:
    pose = loc.locate(timeout_s=2.0)
    if pose is None:
        raise RuntimeError("no cup detected within timeout")
    plan_path_to(pose.position_robot_mm)
```

### Polling in a loop (continuous tracking)

The smoother keeps state across calls, so a second `locate()` is faster and more stable than the first:

```python
with CupLocator("calibration/profiles/session_multitag.json") as loc:
    while not stop_requested:
        pose = loc.locate(timeout_s=0.5)
        if pose is None:
            handle_missed_detection()
            continue
        update_target(pose.position_robot_mm, pose.quality, pose.track_id)
```

### With error handling and quality gating

```python
with CupLocator("calibration/profiles/session_multitag.json") as loc:
    pose = loc.locate(timeout_s=3.0)
    if pose is None:
        return Status.NO_DETECTION
    if pose.quality < 0.4:
        return Status.NOISY_DETECTION
    if pose.height_m is None or pose.height_m < 0.02:
        return Status.OBJECT_TOO_SHORT
    return Status.OK, pose
```

### Restricting the YOLO class (different object)

```python
loc = CupLocator(
    "calibration/profiles/session_multitag.json",
    target_label="bottle",            # YOLO COCO class name
    min_confidence=0.20,
)
```

### Custom YOLO model

```python
loc = CupLocator(
    "calibration/profiles/session_multitag.json",
    model_path="/abs/path/to/my-finetuned-seg.pt",
)
```

### Multiple cups (limitation)

`locate()` returns one cup — the highest-quality stable track. There's currently no `locate_all()`. Workarounds:

- After grasping cup #1, call `locate()` again; the tracker will return the next-best track.
- If you need simultaneous multi-cup awareness, ask the sender to expose `locate_all()`.

---

## 11. Performance & timing

| Event | Typical latency |
|---|---|
| First `CupLocator(...)` construction | < 100 ms (profile load + object construction; no YOLO load yet) |
| First `locate()` call (cold) | 1.5–4 s (YOLO model load + warmup + camera connect + several frames of settling) |
| Subsequent `locate()` calls | 0.3–1.5 s (depends on `settle_frames` and how quickly the tracker re-locks) |
| Per-frame pipeline cost | ~30–60 ms (ChArUco + YOLO + localizer + tracker) |
| Camera frame rate | iPhone Record3D streams ~30 fps; the locator drops intermediate frames if the consumer (you) lags |

Tip: pre-warm the locator at startup:

```python
loc = CupLocator("...")
_ = loc.locate(timeout_s=1.0)   # ignore result; warms up YOLO + camera
# ... later, real calls return faster
```

---

## 12. Tuning for your environment

| Symptom | Knob | Direction |
|---|---|---|
| Detection unreliable / occasional misses | `min_confidence` | Lower (0.05–0.10) |
| Too many spurious detections (wrong objects) | `min_confidence` | Raise (0.30+) |
| First `locate()` returns the cup half-localized | `settle_frames` | Raise (12–15) |
| Demo wants any detection ASAP | `settle_frames` | Lower (2–3) |
| Noisy ChArUco anchor causes coordinate jitter | `anchor_max_reproj_px` | Lower (1.0) |
| ChArUco gets rejected too often in low light | `anchor_max_reproj_px` | Raise (2.5–3.0) |
| Cup detected as `"wine glass"` not `"cup"` | `target_label` | `"wine glass"` (or whatever YOLO emits) |
| Returns very low-quality matches | `min_quality` | Raise (0.30–0.40) |

Run `python -m perception.demos.cup_locator_demo --profile ...` (continuous polling) to watch quality numbers live while you tune.

---

## 13. Troubleshooting

### `FileNotFoundError` on construction
The profile path is wrong. The error message includes the path it tried.

### `ValueError: profile ... has no robot_world_transform`
The profile JSON exists but wasn't fully calibrated. The sender needs to re-run touch-calibration and `save`.

### `ValueError: profile ... has no charuco_board geometry`
Same as above — the board section is missing. Re-touch-cal.

### `locate()` always returns `None`

Walk down the stack:

1. **Camera not delivering frames?** Run `python -m perception.demos.cup_locator_demo --profile ... --max-misses 5`. If you see `no detection` lines but no other errors, the camera is connected but downstream is failing. If you see no lines at all, Record3D isn't streaming.
2. **ChArUco not visible?** The locator silently waits for a board anchor; if the board isn't fully in frame, no detection ever runs. Move the camera or board.
3. **YOLO labels it differently?** Try `target_label="wine glass"`, `"bowl"`, `"bottle"`, `"vase"`. Or run `python -m perception.demos.yolo_world_monitor` to see what labels YOLO is emitting on every frame.
4. **Settle gate too strict?** Lower to `settle_frames=2, min_confidence=0.05` temporarily; if that returns a pose, the issue is detection stability, not absence.

### Detections appear, but coordinates look wrong

- **Check `position_world_m` first**: that's the board-frame number, independent of `T_robot_world`. If world coordinates make sense (the cup at the centre of the board reports ~`(0.1, 0.07, -0.025)` for a 10×7 20 mm board with a 5 cm cup), but robot coordinates don't, the calibration is stale — the robot or board has moved. Ask the sender to recalibrate.
- **If `position_world_m` itself is wrong**: ChArUco might be detecting an aliased pose (rare with a multi-marker board). Check the `track_id` — if it's changing every call, the tracker is dropping the cup and a new ID gets assigned each time, suggesting unstable detection rather than a calibration bug.

### `height_m` reads short (e.g. 20 mm on a 50 mm cup)
This is a known measurement bias when YOLO's mask covers the cup's open top — depth samples are dragged toward the interior. The XY localisation is still correct; only the height is biased. Discussed in the sender's calibration notes.

### `pymycobot` import fails
Install it: `pip install pymycobot`. Even if you don't use the MyCobot driver, the package is pulled in by `world_to_robot`'s transitive import. (If this becomes an annoyance, ask the sender to refactor `cup_locator/api.py` to lazy-import.)

---

## 14. Recalibrating

You shouldn't have to. The cases where you would:

| Change | Action |
|---|---|
| Camera moved (tripod nudged) | Nothing — the live ChArUco anchor handles it per frame. |
| Robot base or ChArUco board moved | Recalibrate (see below). The relationship between robot and board is what `T_robot_world` encodes. |
| Different printed ChArUco board | Recalibrate with the new geometry. The board spec is baked into the saved profile. |
| Different cup / different object | No recalibration; pass `target_label=...` in the constructor. |

To recalibrate, either:

**A.** Run the CLI directly (recommended; you can interact with the touch prompt):

```bash
python -m perception.cup_locator.calibrate \
    --port /dev/ttyUSB0 \
    --profile calibration/profiles/session_multitag.json \
    --squares-x 11 --squares-y 8 \
    --square-mm 20 --marker-mm 14 \
    --dict DICT_4X4_50 \
    --tip-offset-mm 12
```

See `CALIBRATION.md` for the prompt commands (`TL/TR/BL/BR`, `col,row`, `free`, `lock`, `fit`, `save`, ...).

**B.** Programmatic, from inside your code:

```python
with CupLocator("calibration/profiles/session_multitag.json") as loc:
    loc.recalibrate(
        port="/dev/ttyUSB0",
        squares_x=11, squares_y=8,
        square_mm=20, marker_mm=14,
        dict="DICT_4X4_50",
        tip_offset_mm=12,
    )
    # profile is now reloaded; new locate() calls use the fresh transform
    pose = loc.locate(timeout_s=2.0)
```

---

## 15. Limitations & non-goals

- **One cup at a time.** No `locate_all()` yet.
- **One camera at a time.** Record3D / iPhone only. Swap requires changing `CupLocator`'s internals (the `Record3DSource` is hardcoded).
- **No closed-loop visual servoing.** The locator gives you snapshots; if the cup moves between your `locate()` and your grasp, the snapshot is stale.
- **`height_m` is biased low for open-top cups** — see Troubleshooting.
- **No reach-checking.** The locator returns coordinates; whether the MyCobot can actually reach them is your path planner's problem.
- **No grasp planning.** Approach direction, gripper orientation, and force are all yours. The locator gives you "where the cup is," nothing more.

---

## 16. Quick reference card

```python
from perception.cup_locator import CupLocator

with CupLocator("calibration/profiles/session_multitag.json") as loc:
    pose = loc.locate(timeout_s=2.0)
    if pose is None:
        raise RuntimeError("no cup detected")
    x_mm, y_mm, z_mm = pose.position_robot_mm
```

| Need | Look at |
|---|---|
| Where is the cup? | `pose.position_robot_mm` |
| How confident? | `pose.quality` (0..1) |
| How tall is it? | `pose.height_m` (smoothed) |
| Is this the same cup as last call? | `pose.track_id` |
| Detect a different object | `CupLocator(..., target_label="bottle")` |
| Faster first detection | `CupLocator(..., settle_frames=3, min_confidence=0.10)` |
| Calibration changed | See [§14 Recalibrating](#14-recalibrating) |
| Interactive prompt commands | See `CALIBRATION.md` |
| Continuous-stream diagnostic | `python -m perception.demos.cup_locator_demo --profile ...` |
| Detection + arm hover in one shot | `python -m perception.demos.goto_cup --profile ...` |

That's the whole API.
