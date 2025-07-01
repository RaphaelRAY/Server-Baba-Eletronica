import cv2


class VideoProcessor:
    """Simple processor that returns frames from the camera."""

    def __init__(self, camera, fps: int = 10):
        self.camera = camera
        self.fps = fps

    def get_processed_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return None
        # Additional processing could be placed here
        return frame
