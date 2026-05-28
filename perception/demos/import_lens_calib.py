"""Copy the output of `calibrate_lens.py` into a calibration profile.

`calibrate_lens.py` writes its result to a standalone JSON
(`calibration/profiles/lens_calib.json` by default) so the calibration step
itself never mutates a profile. This helper takes that JSON and writes it into
the profile's `lens_calibration` field. Production code that already loads the
profile will see the lens calibration automatically the next time it runs.

The other team's code is unaffected: the new field is optional, both in the
dataclass (Optional[LensCalibration] = None) and in the JSON loader
(silently absent → field stays None → downstream code falls back to pinhole).

    python -m perception.demos.import_lens_calib \\
        --profile calibration/profiles/session_multitag.json \\
        --lens-calib calibration/profiles/lens_calib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from perception.calibration import (
    CalibrationProfile,
    CalibrationProfileIO,
    LensCalibration,
)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m perception.demos.import_lens_calib")
    p.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    p.add_argument("--lens-calib", default="calibration/profiles/lens_calib.json")
    args = p.parse_args()

    profile_path = Path(args.profile)
    lens_path = Path(args.lens_calib)
    if not profile_path.exists():
        print(f"ERROR: profile not found: {profile_path}", file=sys.stderr); sys.exit(2)
    if not lens_path.exists():
        print(f"ERROR: lens calib JSON not found: {lens_path}", file=sys.stderr); sys.exit(2)

    profile: CalibrationProfile = CalibrationProfileIO.load(str(profile_path))
    raw = json.loads(lens_path.read_text())

    # Map the calibrate_lens.py JSON shape to LensCalibration fields. Drop
    # any keys that aren't part of the dataclass (e.g. cells_covered) so we
    # can tolerate future calibrator additions without churning the schema.
    fields = {
        "k_refined": raw["k_refined"],
        "dist_coeffs": raw["dist_coeffs"],
        "image_size_wh": raw["image_size_wh"],
        "rms_reproj_px": float(raw["rms_reproj_px"]),
        "n_captures": int(raw["n_captures"]),
        "captured_at": str(raw.get("captured_at", "")),
    }
    lc = LensCalibration(**fields)

    if profile.lens_calibration is not None:
        prev = profile.lens_calibration
        print(f"  replacing existing lens_calibration "
              f"(rms={prev.rms_reproj_px:.3f}px from {prev.captured_at}).")
    profile.set_lens_calibration(lc)
    CalibrationProfileIO.save(profile, str(profile_path))

    k = lc.k_array()
    d = lc.dist_array()
    print(f"\nwrote lens_calibration into {profile_path}")
    print(f"  K_refined: fx={k[0,0]:.2f}  fy={k[1,1]:.2f}  "
          f"cx={k[0,2]:.2f}  cy={k[1,2]:.2f}  "
          f"(W,H)=({lc.image_size_wh[0]},{lc.image_size_wh[1]})")
    print(f"  dist:     k1={d[0]:+.4f}  k2={d[1]:+.4f}  "
          f"p1={d[2]:+.4f}  p2={d[3]:+.4f}  k3={d[4]:+.4f}")
    print(f"  RMS reproj: {lc.rms_reproj_px:.3f} px over {lc.n_captures} captures.")
    print("\nDownstream effect:")
    print("  - `localize_objects_rgbd(..., lens_calibration=profile.lens_calibration)` "
          "now uses K_refined + dist_coeffs.")
    print("  - `detect_board_pose(rgb, K, cfg, dist_coeffs=lc.dist_array())` "
          "(call-site update) uses the lens model for PnP.")
    print("  - Callers that don't opt in are unchanged (pinhole, no distortion).")


if __name__ == "__main__":
    main()
