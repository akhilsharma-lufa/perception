# perception

Clean-start perception library for iPhone Record3D + AprilTag + robot-ready outputs.

## Current scope

- Frame ingestion from Record3D.
- Geometry utilities (`SE(3)` transforms, camera model helpers, transform tree).
- Boardless multi-tag calibration with fixed origin tag support.
- Automated session calibration manager with profile persistence.
- Runtime anchor estimation with quality metrics and fallback events.
- Motion-aware replan policy primitives for robot integration.
- YOLO detection support (`seg` default, optional `obb`) with RGB-D localization.

Detailed math and stability reference:
- `perception/perception_math_and_stability_guide.html`

## Quick start

1. Ensure dependencies are installed:
   - `record3d`
   - `pupil-apriltags`
   - `numpy`
   - `opencv-python`

2. Run automatic calibration session:

```bash
python -m perception.demos.auto_calibrate_tags
```

This writes calibration output to:

- `calibration/profiles/session_multitag.json`

3. Run live monitor (RGB + depth + confidence + anchor health):

```bash
python -m perception.demos.live_monitor --profile calibration/profiles/session_multitag.json
```

Single-window tiled dashboard is the default. To log stability metrics:

```bash
python -m perception.demos.live_monitor \
  --profile calibration/profiles/session_multitag.json \
  --log-csv logs/perception_live.csv
```

Dashboard includes a dedicated **World Coordinates** section (top-down map in `tag1` frame), so coordinate labels are separated from RGB.

Live monitor controls:

- `q` or `esc`: quit
- `o`: cycle portrait/landscape orientations
- `p`: portrait
- `l`: landscape clockwise
- `k`: landscape counter-clockwise
- `v`: toggle tiled dashboard / separate windows
- `a`: toggle world `X/Y` axes (origin at tag 1)
- `g`: toggle perspective world grid with numeric `x,y` labels (meters)
- `r`: re-run session calibration and refresh world-tag map

Optional flags:

- `--separate-windows`: start with three windows instead of tiled dashboard
- `--log-every-n-frames N`: reduce CSV logging frequency

4. Run YOLO world monitor (cups, position/yaw/height):

```bash
python -m perception.demos.yolo_world_monitor \
  --profile calibration/profiles/session_multitag.json \
  --yolo-model yolo26n-seg.pt \
  --yolo-mode seg
```

For OBB experiments:

```bash
python -m perception.demos.yolo_world_monitor \
  --profile calibration/profiles/session_multitag.json \
  --yolo-model yolo26n-obb.pt \
  --yolo-mode obb
```

Notes:
- World axes/grid overlays on RGB are off by default to reduce visual clutter.
- World Coordinates panel shows mapped tag coordinates and highlights visible tags.

## World frame policy

- `tag 1` is the default world origin (`origin_tag_id=1`).
- Additional tags improve stability and occlusion resilience.
- Runtime anchor events:
  - `ANCHOR_HOLD_LAST`
  - `ANCHOR_LOST`
  - `ANCHOR_UNSTABLE`

## Resilience notes (camera movement)

- Small camera micro-movements are tolerated when multiple mapped tags remain visible.
- If mapped tags disappear briefly, the anchor can hold last pose for a short window.
- If tag agreement degrades, `ANCHOR_UNSTABLE` is emitted and quality drops.
- If world anchor is lost, `ANCHOR_LOST` is emitted and robot-facing updates should pause.

## Synchronization and confidence usage

- Record3D callback captures `rgb`, `depth`, `confidence`, intrinsics, and pose in one packet, and `perception` processes that packet as a synchronized unit.
- Depth confidence is actively used in YOLO RGB-D localization (`confidence >= confidence_floor`) to reject low-trust depth pixels before 3D estimation.
- Camera pose is ingested but currently not fused into world-anchor math; AprilTag multi-tag anchoring remains the primary absolute reference.
