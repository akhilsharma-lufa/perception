import time

import cv2
import numpy as np

from .models import DEVICE_TYPE__TRUEDEPTH, FramePacket


class OpenCVRenderer:
    def __init__(self, target_render_fps: float = 20.0, contour_every_n_frames: int = 2):
        self.target_render_fps = max(1.0, float(target_render_fps))
        self.contour_every_n_frames = max(1, int(contour_every_n_frames))

        self.use_depth_colormap = True
        self.show_depth_contours = True
        self.show_distance_text = True
        self.show_hud = True
        self.show_confidence = True

        self.mouse_x = None
        self.mouse_y = None
        self.prev_render_ts = None
        self.render_fps = 0.0
        self.render_frame_counter = 0

    def setup_windows(self):
        cv2.namedWindow("Depth")
        cv2.setMouseCallback("Depth", self._on_mouse_depth)
        print("Controls: q/esc=quit, c=contours, n=distance text, h=HUD, m=colormap, f=confidence")

    @staticmethod
    def _safe_depth_at(depth: np.ndarray, x: int, y: int):
        h, w = depth.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        val = float(depth[y, x])
        if not np.isfinite(val) or val <= 0:
            return None
        return val

    def _on_mouse_depth(self, event, x, y, flags, param):
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_MOUSELEAVE:
            self.mouse_x = None
            self.mouse_y = None

    def _build_depth_visualization(self, depth: np.ndarray, draw_contours: bool):
        depth_float = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth_float) & (depth_float > 0)

        if not np.any(valid):
            depth_u8 = np.zeros(depth_float.shape, dtype=np.uint8)
            return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        near = float(np.percentile(depth_float[valid], 2))
        far = float(np.percentile(depth_float[valid], 98))
        if far <= near:
            far = near + 1e-3

        depth_u8 = ((np.clip(depth_float, near, far) - near) / (far - near) * 255).astype(
            np.uint8
        )

        if self.use_depth_colormap:
            depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        else:
            depth_vis = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        if self.show_depth_contours and draw_contours:
            for level in (32, 64, 96, 128, 160, 192, 224):
                mask = cv2.inRange(depth_u8, level - 2, level + 2)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    cv2.drawContours(depth_vis, contours, -1, (0, 0, 0), 1)
        return depth_vis

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
            cv2.putText(
                depth_vis,
                f"({mx},{my}) {mouse_depth:.3f} m",
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

    def _draw_hud(self, rgb: np.ndarray, packet: FramePacket, frames_dropped: int):
        if not self.show_hud:
            return
        latency_ms = (time.monotonic() - packet.ts_monotonic) * 1000.0
        hud = (
            f"frame={packet.frame_id} fps={self.render_fps:.1f} lat={latency_ms:.1f}ms "
            f"drop={frames_dropped} dev={packet.device_type} target={self.target_render_fps:.1f}"
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

    @staticmethod
    def _draw_detection_overlays(rgb: np.ndarray, packet: FramePacket):
        for ov in packet.overlays:
            mask = np.asarray(ov.mask).astype(bool)
            if mask.shape[:2] != rgb.shape[:2]:
                continue

            # Alpha-blend segmentation region on RGB image.
            color = np.array([0, 200, 255], dtype=np.uint8)
            overlay = rgb.copy()
            overlay[mask] = (0.6 * overlay[mask] + 0.4 * color).astype(np.uint8)
            rgb[:] = overlay

            # Draw mask contour boundary for object separation.
            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cv2.drawContours(rgb, contours, -1, (0, 255, 255), 2)

            dist_txt = "n/a" if ov.distance_m is None else f"{ov.distance_m:.3f}m"
            txt = f"{ov.label} {ov.confidence:.2f} {dist_txt}"
            txt_origin = (ov.anchor_x, max(18, ov.anchor_y - 8))
            cv2.putText(
                rgb,
                txt,
                txt_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
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

    def render(self, packet: FramePacket, frames_dropped: int) -> bool:
        depth = packet.depth
        rgb = packet.rgb
        confidence = packet.confidence

        self.render_frame_counter += 1
        draw_contours = self.render_frame_counter % self.contour_every_n_frames == 0

        if packet.device_type == DEVICE_TYPE__TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_vis = self._build_depth_visualization(depth, draw_contours)
        self._draw_distance_overlay(rgb, depth_vis, depth)
        self._update_fps()
        self._draw_hud(rgb, packet, frames_dropped)
        self._draw_detection_overlays(rgb, packet)

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
