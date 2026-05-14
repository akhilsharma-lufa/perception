from r3d_sync import SyncApp


if __name__ == "__main__":
    app = SyncApp(queue_size=4, target_render_fps=20.0, contour_every_n_frames=2)
    app.run(device_index=0)
