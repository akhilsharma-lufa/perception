from collections import deque
from dataclasses import dataclass
from threading import Event, Lock
import time

import cv2
import numpy as np
from record3d import Record3DStream


DEVICE_TYPE__TRUEDEPTH = 0
DEVICE_TYPE__LIDAR = 1


@dataclass
class FramePacket:
    frame_id: int
    ts_monotonic: float
    rgb: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    intrinsic_mat: np.ndarray
    camera_pose: object
    device_type: int


class SyncedRecord3D:
    def __init__(
        self,
        queue_size: int = 4,
        target_render_fps: float = 20.0,
        contour_every_n_frames: int = 2,
    ):
        self.session = Record3DStream()
        self.event = Event()
        self.lock = Lock()
        self.queue = deque(maxlen=queue_size)
        self.target_render_fps = max(1.0, float(target_render_fps))
        self.render_period_s = 1.0 / self.target_render_fps
        self.contour_every_n_frames = max(1, int(contour_every_n_frames))

        self.frame_counter = 0
        self.frames_enqueued = 0
        self.frames_dropped = 0
        self.last_stats_log_ts = time.monotonic()
        self.prev_render_ts = None
        self.render_fps = 0.0
        self.render_frame_counter = 0

        # Visualization controls
        self.use_depth_colormap = True
        self.show_depth_contours = True
        self.show_distance_text = True
        self.show_hud = True
        self.show_confidence = True

        # Mouse hover on depth view
        self.mouse_x = None
        self.mouse_y = None

    @staticmethod
    def get_intrinsic_mat_from_coeffs(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]],
            dtype=np.float32,
        )

    def on_new_frame(self):
        """
        Capture callback from a non-main thread.
        Keep this minimal: copy data and queue a coherent packet.
        """
        ts = time.monotonic()
        frame_id = self.frame_counter
        self.frame_counter += 1

        depth = self.session.get_depth_frame().copy()
        rgb = self.session.get_rgb_frame().copy()
        confidence = self.session.get_confidence_frame().copy()
        intrinsic_mat = self.get_intrinsic_mat_from_coeffs(
            self.session.get_intrinsic_mat()
        )
        camera_pose = self.session.get_camera_pose()
        device_type = self.session.get_device_type()

        packet = FramePacket(
            frame_id=frame_id,
            ts_monotonic=ts,
            rgb=rgb,
            depth=depth,
            confidence=confidence,
            intrinsic_mat=intrinsic_mat,
            camera_pose=camera_pose,
            device_type=device_type,
        )

        with self.lock:
            if len(self.queue) == self.queue.maxlen:
                self.frames_dropped += 1
            self.queue.append(packet)
            self.frames_enqueued += 1

        self.event.set()

    def on_stream_stopped(self):
        print("stream stopped")

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

    def _pop_latest_packet(self):
        with self.lock:
            if not self.queue:
                return None
            latest = self.queue[-1]
            self.queue.clear()
            return latest

    def _on_mouse_depth(self, event, x, y, flags, param):
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_MOUSELEAVE:
            self.mouse_x = None
            self.mouse_y = None

    @staticmethod
    def _safe_depth_at(depth: np.ndarray, x: int, y: int):
        h, w = depth.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        val = float(depth[y, x])
        if not np.isfinite(val) or val <= 0:
            return None
        return val

    def _build_depth_visualization(self, depth: np.ndarray, draw_contours: bool):
        depth_float = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth_float) & (depth_float > 0)

        if not np.any(valid):
            depth_u8 = np.zeros(depth_float.shape, dtype=np.uint8)
            depth_vis = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
            return depth_vis, depth_u8

        near = float(np.percentile(depth_float[valid], 2))
        far = float(np.percentile(depth_float[valid], 98))
        if far <= near:
            far = near + 1e-3

        depth_clipped = np.clip(depth_float, near, far)
        depth_norm = (depth_clipped - near) / (far - near)
        depth_u8 = (depth_norm * 255).astype(np.uint8)

        if self.use_depth_colormap:
            depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        else:
            depth_vis = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        if self.show_depth_contours and draw_contours:
            levels = (32, 64, 96, 128, 160, 192, 224)
            for level in levels:
                mask = cv2.inRange(depth_u8, level - 2, level + 2)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    cv2.drawContours(depth_vis, contours, -1, (0, 0, 0), 1)

        return depth_vis, depth_u8

    def _draw_distance_overlay(
        self, rgb: np.ndarray, depth_vis: np.ndarray, depth: np.ndarray
    ):
        if not self.show_distance_text:
            return

        h, w = depth.shape[:2]
        cx = w // 2
        cy = h // 2

        center_depth = self._safe_depth_at(depth, cx, cy)
        cv2.drawMarker(
            depth_vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA
        )
        if center_depth is not None:
            txt = f"center: {center_depth:.3f} m"
            cv2.putText(
                depth_vis,
                txt,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                txt,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.mouse_x is None or self.mouse_y is None:
            return
        mx = int(self.mouse_x)
        my = int(self.mouse_y)
        mouse_depth = self._safe_depth_at(depth, mx, my)
        cv2.circle(depth_vis, (mx, my), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if mouse_depth is not None:
            mouse_txt = f"({mx},{my}) {mouse_depth:.3f} m"
            cv2.putText(
                depth_vis,
                mouse_txt,
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _update_fps(self):
        now = time.monotonic()
        if self.prev_render_ts is not None:
            dt = now - self.prev_render_ts
            if dt > 0:
                instant_fps = 1.0 / dt
                self.render_fps = 0.90 * self.render_fps + 0.10 * instant_fps
        self.prev_render_ts = now

    def _draw_hud(self, rgb: np.ndarray, packet: FramePacket):
        if not self.show_hud:
            return
        latency_ms = (time.monotonic() - packet.ts_monotonic) * 1000.0
        hud = (
            f"frame={packet.frame_id} fps={self.render_fps:.1f} "
            f"lat={latency_ms:.1f}ms drop={self.frames_dropped} dev={packet.device_type} "
            f"target={self.target_render_fps:.1f}"
        )
        cv2.putText(
            rgb,
            hud,
            (10, rgb.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 255, 30),
            1,
            cv2.LINE_AA,
        )

    def _handle_key(self, key: int):
        if key in (27, ord("q")):
            return False
        if key == ord("c"):
            self.show_depth_contours = not self.show_depth_contours
            print(f"[vis] contours={self.show_depth_contours}")
        elif key == ord("n"):
            self.show_distance_text = not self.show_distance_text
            print(f"[vis] distances={self.show_distance_text}")
        elif key == ord("h"):
            self.show_hud = not self.show_hud
            print(f"[vis] hud={self.show_hud}")
        elif key == ord("m"):
            self.use_depth_colormap = not self.use_depth_colormap
            print(f"[vis] colormap={self.use_depth_colormap}")
        elif key == ord("f"):
            self.show_confidence = not self.show_confidence
            print(f"[vis] confidence={self.show_confidence}")
        return True

    def _render_packet(self, packet: FramePacket):
        depth = packet.depth
        rgb = packet.rgb
        confidence = packet.confidence

        self.render_frame_counter += 1
        draw_contours = (
            self.render_frame_counter % self.contour_every_n_frames == 0
        )

        if packet.device_type == DEVICE_TYPE__TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_vis, _ = self._build_depth_visualization(depth, draw_contours)
        self._draw_distance_overlay(rgb, depth_vis, depth)
        self._update_fps()
        self._draw_hud(rgb, packet)

        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth_vis)

        if self.show_confidence and confidence.shape[0] > 0 and confidence.shape[1] > 0:
            conf_vis = np.squeeze(np.asarray(confidence, dtype=np.float32))
            if conf_vis.ndim == 2:
                conf_vis = cv2.normalize(
                    conf_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                cv2.imshow("Confidence", conf_vis)

        key = cv2.waitKey(1) & 0xFF
        return self._handle_key(key)

    def _log_stats(self, packet: FramePacket):
        now = time.monotonic()
        if now - self.last_stats_log_ts < 1.0:
            return

        latency_ms = (now - packet.ts_monotonic) * 1000.0
        pose = packet.camera_pose
        print(
            f"[sync] frame={packet.frame_id} device={packet.device_type} "
            f"queue_drop={self.frames_dropped} latency_ms={latency_ms:.1f} "
            f"pose_t=({pose.tx:.3f}, {pose.ty:.3f}, {pose.tz:.3f})"
        )
        self.last_stats_log_ts = now

    def start_processing_stream(self):
        cv2.namedWindow("Depth")
        cv2.setMouseCallback("Depth", self._on_mouse_depth)
        print("Controls: q/esc=quit, c=contours, n=distance text, h=HUD, m=colormap, f=confidence")
        latest_packet = None
        next_render_ts = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                timeout = max(0.0, min(0.25, next_render_ts - now))
                self.event.wait(timeout=timeout)
                packet = self._pop_latest_packet()
                self.event.clear()

                if packet is not None:
                    latest_packet = packet

                now = time.monotonic()
                if latest_packet is None or now < next_render_ts:
                    continue

                self._log_stats(latest_packet)
                keep_running = self._render_packet(latest_packet)
                if not keep_running:
                    break

                next_render_ts += self.render_period_s
                if now - next_render_ts > (3.0 * self.render_period_s):
                    next_render_ts = now + self.render_period_s
        finally:
            self.session.disconnect()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SyncedRecord3D(queue_size=4, target_render_fps=20.0, contour_every_n_frames=2)
    app.connect_to_device(device_index=0)
    app.start_processing_stream()
