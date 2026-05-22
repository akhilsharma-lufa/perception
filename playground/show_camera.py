"""Read a USB camera and display the live video. Press 'q' to quit."""

import sys
import time

import cv2


def main(camera_index: int = 0) -> int:
    # AVFoundation is the native macOS backend; being explicit avoids fallback weirdness.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}")
        return 1

    # Many USB cameras emit blank/green frames for the first ~1s while exposure settles.
    print("Warming up...")
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Streaming camera {camera_index}: {w}x{h} @ {fps:.1f} fps. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame; exiting.")
                break

            cv2.imshow("USB Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sys.exit(main(index))
