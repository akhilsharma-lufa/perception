from r3d_sync import SyncApp
from r3d_sync.processors import YoloDistanceProcessor
from r3d_sync.processors.yolo_distance import YoloSettings


if __name__ == "__main__":
    app = SyncApp(
        queue_size=4,
        target_render_fps=20.0,
        contour_every_n_frames=2,
    )

    # YOLO inference is intentionally throttled to keep visualization responsive.
    # Use a segmentation model so overlays are masks, not bounding boxes.
    yolo_processor = YoloDistanceProcessor(
        YoloSettings(
            model_path="yolo26n-seg.pt",
            min_confidence=0.35,
            inference_every_n_frames=2,
            confidence_floor=1,
        )
    )
    app.add_packet_processor(yolo_processor)
    app.run(device_index=0)
