from threading import Event
from typing import Callable, List
import time

import cv2

from .capture import Record3DCapture
from .frame_buffer import LatestFrameBuffer
from .models import FramePacket
from .renderer import OpenCVRenderer


class SyncApp:
    def __init__(
        self,
        queue_size: int = 4,
        target_render_fps: float = 20.0,
        contour_every_n_frames: int = 2,
    ):
        self.frame_event = Event()
        self.buffer = LatestFrameBuffer(maxlen=queue_size)
        self.capture = Record3DCapture(frame_buffer=self.buffer, frame_event=self.frame_event)
        self.renderer = OpenCVRenderer(
            target_render_fps=target_render_fps,
            contour_every_n_frames=contour_every_n_frames,
        )
        self.target_render_fps = max(1.0, float(target_render_fps))
        self.render_period_s = 1.0 / self.target_render_fps
        self.last_stats_log_ts = time.monotonic()

        # Extension point for YOLO/AprilTags/PCA/orientation/transforms.
        self.packet_processors: List[Callable[[FramePacket], FramePacket]] = []

    def add_packet_processor(self, processor: Callable[[FramePacket], FramePacket]):
        self.packet_processors.append(processor)

    def _run_processors(self, packet: FramePacket) -> FramePacket:
        for processor in self.packet_processors:
            packet = processor(packet)
        return packet

    def _log_stats(self, packet: FramePacket):
        now = time.monotonic()
        if now - self.last_stats_log_ts < 1.0:
            return
        latency_ms = (now - packet.ts_monotonic) * 1000.0
        pose = packet.camera_pose
        print(
            f"[sync] frame={packet.frame_id} device={packet.device_type} "
            f"queue_drop={self.buffer.stats.frames_dropped} latency_ms={latency_ms:.1f} "
            f"pose_t=({pose.tx:.3f}, {pose.ty:.3f}, {pose.tz:.3f})"
        )
        self.last_stats_log_ts = now

    def run(self, device_index: int = 0):
        self.capture.connect_to_device(device_index=device_index)
        self.renderer.setup_windows()
        latest_packet = None
        next_render_ts = time.monotonic()

        try:
            while True:
                now = time.monotonic()
                timeout = max(0.0, min(0.25, next_render_ts - now))
                self.frame_event.wait(timeout=timeout)
                packet = self.buffer.pop_latest()
                self.frame_event.clear()

                if packet is not None:
                    latest_packet = packet

                now = time.monotonic()
                if latest_packet is None or now < next_render_ts:
                    continue

                latest_packet = self._run_processors(latest_packet)
                self._log_stats(latest_packet)
                keep_running = self.renderer.render(
                    latest_packet, frames_dropped=self.buffer.stats.frames_dropped
                )
                if not keep_running:
                    break

                next_render_ts += self.render_period_s
                if now - next_render_ts > (3.0 * self.render_period_s):
                    next_render_ts = now + self.render_period_s
        finally:
            self.capture.disconnect()
            cv2.destroyAllWindows()
