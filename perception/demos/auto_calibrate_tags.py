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
                min_unique_tags=2,
                profile_path="calibration/profiles/session_multitag.json",
            ),
        )
        data = manager.collect_calibration_data(source)
        profile = manager.build_and_save_profile(data)
        print("[perception] Calibration profile saved.")
        print(f"[perception] Origin tag: {profile.origin_tag_id}")
        print(f"[perception] Tags mapped: {len(profile.world_tag_transforms)}")
        print(f"[perception] Metrics: {profile.metrics}")
        if profile.table_plane is not None:
            n = profile.table_plane.normal_world
            o = profile.table_plane.origin_world
            print(
                f"[perception] Table plane fit: normal=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}) "
                f"origin=({o[0]:+.3f},{o[1]:+.3f},{o[2]:+.3f}) "
                f"inlier_ratio={profile.table_plane.inlier_ratio:.2f} "
                f"residual={profile.table_plane.mean_abs_residual_m*1000:.1f} mm"
            )
        else:
            print("[perception] Table plane fit: SKIPPED (insufficient data)")
    finally:
        source.disconnect()


if __name__ == "__main__":
    main()
