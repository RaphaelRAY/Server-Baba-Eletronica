import cv2
from onvif import ONVIFCamera
from threading import Thread, Event


class CameraHandler:
    """Connects to an ONVIF camera and continuously grabs frames."""

    def __init__(self, host: str, port: int, user: str, passwd: str):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self._camera = None
        self._cap = None
        self._thread = None
        self._stop = Event()
        self._frame = None

    def start(self) -> None:
        """Initializes ONVIF and starts capture thread."""
        if self._thread and self._thread.is_alive():
            return
        self._camera = ONVIFCamera(self.host, self.port, self.user, self.passwd)
        media_service = self._camera.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token
        stream_setup = {
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            },
            'ProfileToken': token
        }
        uri = media_service.GetStreamUri(stream_setup).Uri
        # Add credentials to RTSP URL
        if uri.startswith('rtsp://'):
            uri = uri.replace('rtsp://', f'rtsp://{self.user}:{self.passwd}@')
        self._cap = cv2.VideoCapture(uri)
        self._stop.clear()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            if not self._cap:
                break
            ret, frame = self._cap.read()
            if ret:
                self._frame = frame

    def get_frame(self):
        """Returns the latest captured frame."""
        return self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        if self._cap:
            self._cap.release()
