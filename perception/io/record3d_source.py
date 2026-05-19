import time
from threading import Event
from typing import Optional

import numpy as np
from record3d import Record3DStream

from .frame_packet import CameraPose, FramePacket


class Record3DSource:
    """Record3D frame source with latest-packet access."""

    def __init__(self):
        self._session = Record3DStream()
        self._frame_id = 0
        self._latest_packet: Optional[FramePacket] = None
        self._event = Event()

    @staticmethod
    def intrinsic_matrix_from_coeffs(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0.0, coeffs.tx], [0.0, coeffs.fy, coeffs.ty], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _on_new_frame(self):
        packet = FramePacket(
            frame_id=self._frame_id,
            ts_monotonic=time.monotonic(),
            rgb=self._session.get_rgb_frame().copy(),
            depth=self._session.get_depth_frame().copy(),
            confidence=self._session.get_confidence_frame().copy(),
            intrinsic_mat=self.intrinsic_matrix_from_coeffs(self._session.get_intrinsic_mat()),
            camera_pose=CameraPose.from_record3d(self._session.get_camera_pose()),
            device_type=int(self._session.get_device_type()),
        )
        self._frame_id += 1
        self._latest_packet = packet
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
        self._event.clear()
        return packet
