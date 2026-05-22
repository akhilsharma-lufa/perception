"""Probe camera indices 0..9 and show a frame from each so you can identify them.

Press any key to advance to the next camera, or 'q' to quit early.
"""

import cv2


def main() -> None:
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        h, w = frame.shape[:2]
        label = f"index {i}  ({w}x{h})  -- press any key for next, 'q' to quit"
        cv2.putText(
            frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        print(f"camera index {i}: {w}x{h}")
        found.append(i)

        cv2.imshow("camera probe", frame)
        key = cv2.waitKey(0) & 0xFF
        cap.release()
        cv2.destroyAllWindows()
        if key == ord("q"):
            break

    if not found:
        print("No cameras found.")
    else:
        print(f"Available indices: {found}")


if __name__ == "__main__":
    main()
