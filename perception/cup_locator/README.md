# `perception.cup_locator`

A small, self-contained subpackage that hides the perception stack behind a single blocking call: detect one cup, return its coordinates in the **MyCobot 280 base frame (mm)**.

Built so the path-planning side of the project doesn't have to learn ChArUco anchors, YOLO settings, or world-frame tracking. Calibration is owned here too.

## What it does

1. **Calibrate** (interactive, requires the robot) — operator hand-drives the gripper tip onto known ChArUco corners; the script solves `T_robot_world` via Kabsch and writes a profile JSON. Run once, or whenever the robot or board moves.
2. **Locate** (every time the planner needs a cup) — opens the Record3D camera, runs YOLO segmentation + RGB-D percentile localization + world-frame tracker with median height smoothing, and returns when the cup has been seen for several consecutive frames at acceptable quality.

There is no separate "anchor-only" calibration: the live `T_world_camera` is recomputed from every frame inside `locate()`.

## Install

The subpackage only re-uses code that already lives in `perception/`. Drop the whole `perception/` tree into your repo, make sure it is importable, and you are done. Dependencies that must be installed in the consuming environment: `numpy`, `opencv-contrib-python`, `record3d`, `ultralytics`, `pymycobot` (only needed for the calibrate CLI).

## Quickstart

### 1. Create a profile (once, or after the robot/board moves)

```bash
python -m perception.cup_locator.calibrate \
    --port /dev/ttyUSB0 \
    --profile calibration/profiles/session_multitag.json \
    --squares-x 10 --squares-y 7 \
    --square-mm 20 --marker-mm 15 \
    --dict DICT_4X4_50
```

This drops you into a free-drive prompt: move the gripper tip onto a labelled corner, type the corner name (`TL`, `TR`, `BL`, `BR`, or `col,row`), repeat for 3+ corners, then type `fit` and `save`. The board geometry is baked into the saved JSON — you never need to retype it again unless you replace the physical board.

### 2. Smoke-test the locator

```bash
python -m perception.cup_locator \
    --profile calibration/profiles/session_multitag.json
```

Prints one cup pose in robot mm within a few seconds; non-zero exit if nothing stable is detected.

### 3. Use it from code

```python
from perception.cup_locator import CupLocator

with CupLocator("calibration/profiles/session_multitag.json") as loc:
    pose = loc.locate(timeout_s=2.0)
    if pose is None:
        raise RuntimeError("no cup detected")
    x_mm, y_mm, z_mm = pose.position_robot_mm
    # plan_path_to((x_mm, y_mm, z_mm))
```

## API

### `CupPose`

```python
@dataclass(frozen=True)
class CupPose:
    position_robot_mm: tuple[float, float, float]   # primary output
    position_world_m:  tuple[float, float, float]   # board-frame, for debugging
    height_m:          float | None                  # smoothed (median window)
    quality:           float                         # 0..1, smoothed
    yaw_hint_rad:      float | None                  # gripper yaw from mask PCA
    label:             str
    track_id:          str                           # e.g. "cup-3"
```

### `CupLocator(profile_path, *, target_label="cup", min_confidence=0.15, settle_frames=8, min_quality=0.20, anchor_max_reproj_px=1.5, model_path=None)`

- `target_label` — YOLO class to detect. Defaults to `"cup"`.
- `min_confidence` — YOLO confidence floor; lowered from the 0.30 default so faint cups still register.
- `settle_frames` — how many consecutive frames the tracker must see the same cup before `locate()` returns it. Higher → more stable, slower.
- `min_quality` — minimum smoothed quality (0..1) for a track to count toward the settle gate.
- `anchor_max_reproj_px` — reject ChArUco detections worse than this reprojection error.

#### `loc.locate(timeout_s=2.0, settle_frames=None, *, device_index=0) -> CupPose | None`

Blocks. Opens the camera lazily on the first call. Returns the highest-quality track that has passed the settle gate, or `None` on timeout. Subsequent calls re-use the open camera and the world tracker's smoothed state, so a second `locate()` typically returns faster than the first.

#### `loc.recalibrate(**touch_cal_kwargs)`

Invokes the interactive touch-cal flow against `loc`'s profile path, then reloads. Forwards kwargs as `--flag value` (booleans become bare `--flag` / `--no-flag`). Useful when the board or robot has moved mid-session.

#### `loc.close()`

Releases the camera. Idempotent. Also called via the context-manager `__exit__`.

## Coordinate convention

`position_robot_mm` is in the frame returned by `pymycobot.MyCobot.get_coords()` — i.e. the same frame `send_coords` consumes. To grasp from above, pair it with `RPY = (180, 0, 0)`:

```python
x_mm, y_mm, z_mm = pose.position_robot_mm
driver.send_coords_mm_deg([x_mm, y_mm, z_mm, 180.0, 0.0, 0.0], speed=25)
```

`position_world_m` is in the board frame (origin at the ChArUco board's printed origin, +Z into the table by the convention `touch_calibrate.py` saves). Useful for sanity-checking against the visual scene.

## When to re-run calibration

| Change | What to do |
|---|---|
| Camera moved (tripod nudged) | Nothing — the live anchor handles it per frame. |
| Robot base or ChArUco board moved | Re-run `python -m perception.cup_locator.calibrate ...`. The robot↔world transform is now stale. |
| Different printed ChArUco board (different size/dict) | Re-run calibrate with the new `--squares-x/y --square-mm --marker-mm --dict` flags. The new geometry is baked into the profile. |
| Different cup / different YOLO class | No recalibration; just pass `target_label=...` when constructing `CupLocator`. |
