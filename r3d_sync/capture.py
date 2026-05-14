import time
from threading import Event

import numpy as np
from record3d import Record3DStream

from .frame_buffer import LatestFrameBuffer
from .models import FramePacket


class Record3DCapture:
    def __init__(self, frame_buffer: LatestFrameBuffer, frame_event: Event):
        self.session = Record3DStream()
        self.frame_buffer = frame_buffer
        self.frame_event = frame_event
        self.frame_counter = 0

    @staticmethod
    def _intrinsic_matrix_from_coeffs(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]],
            dtype=np.float32,
        )

    def _on_new_frame(self):
        ts = time.monotonic()
        frame_id = self.frame_counter
        self.frame_counter += 1

        packet = FramePacket(
            frame_id=frame_id,
            ts_monotonic=ts,
            rgb=self.session.get_rgb_frame().copy(),
            depth=self.session.get_depth_frame().copy(),
            confidence=self.session.get_confidence_frame().copy(),
            intrinsic_mat=self._intrinsic_matrix_from_coeffs(
                self.session.get_intrinsic_mat()
            ),
            camera_pose=self.session.get_camera_pose(),
            device_type=self.session.get_device_type(),
        )
        self.frame_buffer.push(packet)
        self.frame_event.set()

    @staticmethod
    def _on_stream_stopped():
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

        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stream_stopped
        if not self.session.connect(devices[device_index]):
            raise RuntimeError("Unable to connect to selected device.")

    def disconnect(self):
        self.session.disconnect()
