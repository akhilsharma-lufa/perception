from perception.calibration import (
    AutoCalibrationManager,
    AutoCalibrationSettings,
    MultiTagCalibrator,
    MultiTagCalibratorSettings,
)
from perception.io import Record3DSource


def main():
    source = Record3DSource()
    source.connect(device_index=0)
    try:
        calibrator = MultiTagCalibrator(
            MultiTagCalibratorSettings(
                family="tag36h11",
                tag_size_m=0.04,
                origin_tag_id=1,
            )
        )
        manager = AutoCalibrationManager(
            calibrator=calibrator,
            settings=AutoCalibrationSettings(
                target_frames=140,
                max_collection_seconds=25.0,
                min_unique_tags=3,
                profile_path="calibration/profiles/session_multitag.json",
            ),
        )
        observations = manager.collect_observations(source)
        profile = manager.build_and_save_profile(observations)
        print("[perception] Calibration profile saved.")
        print(f"[perception] Origin tag: {profile.origin_tag_id}")
        print(f"[perception] Tags mapped: {len(profile.world_tag_transforms)}")
        print(f"[perception] Metrics: {profile.metrics}")
    finally:
        source.disconnect()


if __name__ == "__main__":
    main()
