import time
from threading import Event
from typing import Optional

import numpy as np
from record3d import Record3DStream

from .frame_packet import CameraPose, FramePacket


class Record3DSource:
    """Record3D frame source with latest-packet access.

    The callback drops frames when the previous packet hasn't been consumed
    yet, so the C++/iPhone-side queue can drain without our Python callback
    paying the cost of copying every frame. This keeps the visual lag small
    even when our main loop runs slower than the iPhone's capture rate.
    """

    def __init__(self):
        self._session = Record3DStream()
        self._frame_id = 0
        self._latest_packet: Optional[FramePacket] = None
        self._event = Event()
        self._dropped_since_last_log: int = 0
        self._delivered_since_last_log: int = 0
        self._last_drop_log_t: float = time.monotonic()
        # The native (H, W) Record3D's intrinsic matrix is calibrated at — detected
        # lazily on the first frame by which resolution's image centre best matches
        # the matrix's principal point. The matrix is then normalised to RGB
        # resolution before emission, so downstream code (which assumes RGB) is
        # always consistent regardless of what Record3D returns.
        self._intrinsic_native_shape: Optional[tuple] = None

    @staticmethod
    def intrinsic_matrix_from_coeffs(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0.0, coeffs.tx], [0.0, coeffs.fy, coeffs.ty], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _scale_intrinsics(k: np.ndarray, src_shape: tuple, dst_shape: tuple) -> np.ndarray:
        sx = float(dst_shape[1]) / float(src_shape[1])
        sy = float(dst_shape[0]) / float(src_shape[0])
        out = np.array(k, dtype=np.float64, copy=True)
        out[0, 0] *= sx
        out[1, 1] *= sy
        out[0, 2] *= sx
        out[1, 2] *= sy
        return out

    def _resolve_intrinsic_native_shape(
        self, k: np.ndarray, rgb_shape: tuple, depth_shape: tuple
    ) -> tuple:
        """Pick the (H, W) whose image centre best matches the matrix's principal
        point. The principal point sits near the image centre at a sensor's native
        resolution, so this disambiguates Record3D's RGB-vs-depth convention.
        """
        cx, cy = float(k[0, 2]), float(k[1, 2])
        d_rgb = (cx - rgb_shape[1] * 0.5) ** 2 + (cy - rgb_shape[0] * 0.5) ** 2
        d_depth = (cx - depth_shape[1] * 0.5) ** 2 + (cy - depth_shape[0] * 0.5) ** 2
        chosen = rgb_shape if d_rgb <= d_depth else depth_shape
        print(
            f"[record3d] intrinsic matrix native resolution = {chosen[0]}x{chosen[1]} "
            f"(principal point ({cx:.1f},{cy:.1f}); centres "
            f"RGB({rgb_shape[1]/2:.1f},{rgb_shape[0]/2:.1f}) "
            f"depth({depth_shape[1]/2:.1f},{depth_shape[0]/2:.1f})). "
            f"{'No scaling needed.' if chosen == rgb_shape else 'Will scale to RGB before emission.'}"
        )
        return chosen

    def _on_new_frame(self):
        # Drain-but-don't-copy: if the main thread hasn't consumed the previous
        # packet yet, skip this callback to keep the C++ queue draining without
        # paying the per-frame copy cost. We still call _frame_id++ so the
        # downstream loop sees real iPhone-pace frame numbers.
        self._frame_id += 1
        if self._latest_packet is not None:
            self._dropped_since_last_log += 1
            return
        rgb = self._session.get_rgb_frame().copy()
        depth = self._session.get_depth_frame().copy()
        raw_k = self.intrinsic_matrix_from_coeffs(self._session.get_intrinsic_mat())
        rgb_shape = (int(rgb.shape[0]), int(rgb.shape[1]))
        depth_shape = (int(depth.shape[0]), int(depth.shape[1]))
        if self._intrinsic_native_shape is None:
            self._intrinsic_native_shape = self._resolve_intrinsic_native_shape(
                raw_k, rgb_shape, depth_shape
            )
        k_rgb = (raw_k if self._intrinsic_native_shape == rgb_shape
                 else self._scale_intrinsics(raw_k, self._intrinsic_native_shape, rgb_shape))
        packet = FramePacket(
            frame_id=self._frame_id,
            ts_monotonic=time.monotonic(),
            rgb=rgb,
            depth=depth,
            confidence=self._session.get_confidence_frame().copy(),
            intrinsic_mat=k_rgb,
            camera_pose=CameraPose.from_record3d(self._session.get_camera_pose()),
            device_type=int(self._session.get_device_type()),
            intrinsic_shape=self._intrinsic_native_shape,
        )
        self._latest_packet = packet
        self._delivered_since_last_log += 1
        self._event.set()

    @staticmethod
    def _on_stream_stopped():
        print("[perception] stream stopped")

    def connect(self, device_index: int = 0):
        devices = Record3DStream.get_connected_devices()
        if len(devices) <= device_index:
            raise RuntimeError(
                f"Cannot connect to device #{device_index}; found {len(devices)} devices."
            )
        self._session.on_new_frame = self._on_new_frame
        self._session.on_stream_stopped = self._on_stream_stopped
        if not self._session.connect(devices[device_index]):
            raise RuntimeError("Unable to connect to selected Record3D device.")

    def disconnect(self):
        self._session.disconnect()

    def wait_for_frame(self, timeout_s: float = 0.25) -> Optional[FramePacket]:
        self._event.wait(timeout=timeout_s)
        packet = self._latest_packet
        # Mark slot empty so the next callback knows we're ready for a fresh
        # frame; intermediate callbacks while we were processing have been
        # dropped, draining the upstream queue.
        self._latest_packet = None
        self._event.clear()

        now = time.monotonic()
        if now - self._last_drop_log_t >= 2.0:
            delivered = self._delivered_since_last_log
            dropped = self._dropped_since_last_log
            elapsed = now - self._last_drop_log_t
            self._delivered_since_last_log = 0
            self._dropped_since_last_log = 0
            self._last_drop_log_t = now
            total = delivered + dropped
            if total > 0:
                print(
                    f"[record3d] last {elapsed:.1f}s: delivered={delivered} "
                    f"dropped={dropped} (iphone_rate~{total / elapsed:.1f}fps, "
                    f"delivered_rate~{delivered / elapsed:.1f}fps)"
                )
        return packet
