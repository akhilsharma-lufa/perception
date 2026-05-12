from record3d import Record3DStream
import numpy as np
import cv2
from threading import Event

session = Record3DStream()
event = Event()

def on_new_frame():
    # This is called from a non-main thread
    event.set() # Notify the main thread to stop waiting and process the new frame

def on_stream_stopped():
    print("stream stopped")

def connect_to_device(device_index = 0):
    print(f"Searching for devices")
    devices = Record3DStream.get_connected_devices()
    print(f"{len(devices)} devices found:")
    for device in devices:
        print(f"\tID: {device.product_id}, UDID: {device.udid}")

    if len(devices) <= device_index:
        raise RuntimeError(f"Cannot connect to device #{device_index}, try a different index.")
    
    device = devices[device_index]
    session.on_new_frame = on_new_frame
    session.on_stream_stopped = on_stream_stopped
    session.connect(device) # Initialize connection and start capturing

def get_intrinsic_mat_from_coeffs(coeffs):
    return np.array([
        [coeffs.fx, 0,          coeffs.tx],
        [0,         coeffs.fy,  coeffs.ty],
        [0,         0,          1        ]])

DEVICE_TYPE__TRUEDEPTH = 0

DEVICE_TYPE__LIDAR = 1

def start_processing_stream():
    while True:
        event.wait() # wait for the next frame to be ready

        # process the newly arrived RGBD frame
        depth = session.get_depth_frame()
        rgb = session.get_rgb_frame()
        confidence = session.get_confidence_frame()
        intrinsic_mat = get_intrinsic_mat_from_coeffs(session.get_intrinsic_mat())
        camera_pose = session.get_camera_pose() # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

        print(intrinsic_mat)

        # Postprocess it
        if session.get_device_type() == DEVICE_TYPE__TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Show the RGBD Stream
        cv2.imshow('RGB', rgb)
        cv2.imshow('Depth', depth)

        if confidence.shape[0] > 0 and confidence.shape[1] > 0:
            conf_vis = np.squeeze(np.asarray(confidence, dtype=np.float32))
            if conf_vis.ndim == 2:
                conf_vis = cv2.normalize(
                    conf_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                cv2.imshow('Confidence', conf_vis)

        cv2.waitKey(1)

        event.clear()
        

if __name__ == "__main__":
    connect_to_device(device_index = 0)
    start_processing_stream()