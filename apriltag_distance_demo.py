from threading import Event

import cv2
import numpy as np
from pupil_apriltags import Detector
from record3d import Record3DStream


DEVICE_TYPE__TRUEDEPTH = 0


class AprilTagDistanceDemo:
    def __init__(self, tag_family: str = "tag36h11", tag_size_m: float = 0.04):
        self.session = Record3DStream()
        self.event = Event()
        self.detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.5,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.tag_size_m = tag_size_m
        self.required_ids = (3, 1, 2)
        self.smooth_alpha = 0.2
        self._smoothed_w = None
        self._smoothed_h = None

    def on_new_frame(self):
        self.event.set()

    @staticmethod
    def on_stream_stopped():
        print("stream stopped")

    @staticmethod
    def get_intrinsic_mat_from_coeffs(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]],
            dtype=np.float32,
        )

    def connect_to_device(self, device_index: int = 0):
        print("Searching for devices")
        devices = Record3DStream.get_connected_devices()
        print(f"{len(devices)} devices found:")
        for device in devices:
            print(f"\tID: {device.product_id}, UDID: {device.udid}")

        if len(devices) <= device_index:
            raise RuntimeError(
                f"Cannot connect to device #{device_index}, try a different index."
            )

        device = devices[device_index]
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        if not self.session.connect(device):
            raise RuntimeError("Unable to connect to selected device.")

    def run(self):
        print("Press q or Esc to quit.")
        try:
            while True:
                self.event.wait(timeout=0.25)
                self.event.clear()

                rgb = self.session.get_rgb_frame()
                if rgb is None or rgb.size == 0:
                    continue

                rgb = rgb.copy()
                if self.session.get_device_type() == DEVICE_TYPE__TRUEDEPTH:
                    rgb = cv2.flip(rgb, 1)

                intr = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                detections = self.detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=(
                        float(intr[0, 0]),
                        float(intr[1, 1]),
                        float(intr[0, 2]),
                        float(intr[1, 2]),
                    ),
                    tag_size=self.tag_size_m,
                )

                # Sort for stable display order (top-left to bottom-right in image).
                detections = sorted(
                    detections, key=lambda d: (float(d.center[1]), float(d.center[0]))
                )

                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                tags = []
                for idx, det in enumerate(detections):
                    corners = np.asarray(det.corners, dtype=np.int32).reshape(-1, 1, 2)
                    center = tuple(np.asarray(det.center, dtype=np.int32).tolist())
                    pose_t = np.asarray(det.pose_t, dtype=np.float64).reshape(3)
                    pose_r = np.asarray(det.pose_R, dtype=np.float64).reshape(3, 3)

                    cv2.polylines(bgr, [corners], True, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.drawMarker(
                        bgr, center, (255, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        bgr,
                        f"i{idx} id{int(det.tag_id)}",
                        (center[0] + 6, max(18, center[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    tags.append(
                        {
                            "idx": idx,
                            "id": int(det.tag_id),
                            "center": center,
                            "t": pose_t,
                            "r": pose_r,
                        }
                    )

                selected = {}
                for req_id in self.required_ids:
                    for item in tags:
                        if item["id"] == req_id and req_id not in selected:
                            selected[req_id] = item

                def draw_labeled_line(p1, p2, text, color=(0, 255, 255), thickness=2):
                    cv2.line(bgr, p1, p2, color, thickness, cv2.LINE_AA)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    cv2.putText(
                        bgr,
                        text,
                        (mid[0] + 4, mid[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                if all(tag_id in selected for tag_id in self.required_ids):
                    t3 = selected[3]
                    t1 = selected[1]
                    t2 = selected[2]

                    # Pairwise tag-to-tag distance lines (3-1, 1-2, 3-2).
                    for a, b in ((t3, t1), (t1, t2), (t3, t2)):
                        dist_m = float(np.linalg.norm(a["t"] - b["t"]))
                        draw_labeled_line(
                            a["center"],
                            b["center"],
                            f"id{a['id']}-id{b['id']}: {dist_m:.3f} m",
                            color=(0, 255, 255),
                            thickness=2,
                        )

                    # Stable rectangle dimensions in tag-3 frame (camera-orientation invariant).
                    p2_in_3 = t3["r"].T @ (t2["t"] - t3["t"])
                    dx_m_raw = float(abs(p2_in_3[0]))
                    dy_m_raw = float(abs(p2_in_3[1]))
                    if self._smoothed_w is None:
                        self._smoothed_w = dx_m_raw
                        self._smoothed_h = dy_m_raw
                    else:
                        a_ = self.smooth_alpha
                        self._smoothed_w = (1.0 - a_) * self._smoothed_w + a_ * dx_m_raw
                        self._smoothed_h = (1.0 - a_) * self._smoothed_h + a_ * dy_m_raw
                    dx_m = float(self._smoothed_w)
                    dy_m = float(self._smoothed_h)
                    diag_m = float(np.sqrt(dx_m * dx_m + dy_m * dy_m))

                    # Keep on-image right-angle guides for immediate visual intuition.
                    x3, y3 = t3["center"]
                    x2, y2 = t2["center"]
                    a = (x3, y3)
                    b = (x2, y3)
                    c = (x2, y2)
                    d = (x3, y2)

                    # Rectangle sides.
                    draw_labeled_line(a, b, f"top: {dx_m:.3f} m", color=(255, 180, 0), thickness=2)
                    draw_labeled_line(d, c, f"base: {dx_m:.3f} m", color=(255, 180, 0), thickness=2)
                    draw_labeled_line(a, d, f"left: {dy_m:.3f} m", color=(255, 180, 0), thickness=2)
                    draw_labeled_line(b, c, f"right: {dy_m:.3f} m", color=(255, 180, 0), thickness=2)

                    # Both diagonals.
                    draw_labeled_line(a, c, f"diag1: {diag_m:.3f} m", color=(0, 200, 0), thickness=2)
                    draw_labeled_line(b, d, f"diag2: {diag_m:.3f} m", color=(0, 200, 0), thickness=2)

                    # Metric inset (always right-angled) for stable base visualization.
                    inset_x, inset_y = 12, 62
                    inset_w, inset_h = 220, 160
                    cv2.rectangle(
                        bgr,
                        (inset_x, inset_y),
                        (inset_x + inset_w, inset_y + inset_h),
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        bgr,
                        "metric_base(tag3->tag2)",
                        (inset_x + 6, inset_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Fit rectangle preserving aspect ratio.
                    rect_w_px = inset_w - 28
                    rect_h_px = inset_h - 40
                    scale = min(
                        rect_w_px / max(dx_m, 1e-6),
                        rect_h_px / max(dy_m, 1e-6),
                    )
                    rw = int(dx_m * scale)
                    rh = int(dy_m * scale)
                    ox = inset_x + 14
                    oy = inset_y + inset_h - 14
                    pa = (ox, oy)
                    pb = (ox + rw, oy)
                    pc = (ox + rw, oy - rh)
                    pd = (ox, oy - rh)
                    cv2.line(bgr, pa, pb, (255, 180, 0), 2, cv2.LINE_AA)
                    cv2.line(bgr, pb, pc, (255, 180, 0), 2, cv2.LINE_AA)
                    cv2.line(bgr, pc, pd, (255, 180, 0), 2, cv2.LINE_AA)
                    cv2.line(bgr, pd, pa, (255, 180, 0), 2, cv2.LINE_AA)
                    cv2.line(bgr, pa, pc, (0, 200, 0), 2, cv2.LINE_AA)
                    cv2.line(bgr, pb, pd, (0, 200, 0), 2, cv2.LINE_AA)
                else:
                    missing = [str(i) for i in self.required_ids if i not in selected]
                    cv2.putText(
                        bgr,
                        f"Missing required tags: {', '.join(missing)}",
                        (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    bgr,
                    f"tags_detected={len(tags)} required_ids=3,1,2 tag_size={self.tag_size_m:.3f}m alpha={self.smooth_alpha:.2f}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("AprilTag Distance Demo", bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            self.session.disconnect()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AprilTagDistanceDemo(tag_family="tag36h11", tag_size_m=0.04)
    app.connect_to_device(device_index=0)
    app.run()
