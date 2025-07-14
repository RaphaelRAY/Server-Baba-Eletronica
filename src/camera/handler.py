import socket
import os
# Suprime logs do OpenCV/FFmpeg
env = os.environ
env["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
from onvif import ONVIFCamera
from onvif.exceptions import ONVIFError
from threading import Thread, Event, Lock
import time

# Timeout global para conexões socket
socket.setdefaulttimeout(5)

class CameraHandler:
    """Conecta a uma câmera ONVIF e captura/preview de vídeo com baixa latência usando thread única."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        passwd: str,
        width: int = 640,
        height: int = 480
    ):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.width = width
        self.height = height
        self._camera = None
        self._cap = None
        self._thread: Thread = None
        self._stop = Event()
        self._frame = None
        self._lock = Lock()

    def start(self) -> None:
        """Inicializa ONVIF, abre o stream RTSP e inicia thread de captura."""
        # Se já está capturando, ignora
        if self._cap and self._cap.isOpened() and self._thread and self._thread.is_alive():
            return

        try:
            # Inicializa ONVIF
            self._camera = ONVIFCamera(self.host, self.port, self.user, self.passwd)
            media = self._camera.create_media_service()
            profile = media.GetProfiles()[0]
            uri = media.GetStreamUri({
                'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                'ProfileToken': profile.token
            }).Uri
            if uri.startswith('rtsp://'):
                uri = uri.replace('rtsp://', f'rtsp://{self.user}:{self.passwd}@')

            # Abre captura com FFmpeg
            self._cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._cap.isOpened():
                raise RuntimeError(f"Falha ao abrir stream: {uri}")

            # Inicia thread de captura
            self._stop.clear()
            self._thread = Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        except Exception as e:
            print(f"[Erro] Falha ao iniciar câmera: {e}")

    def _capture_loop(self):
        """Loop que faz leitura contínua e armazena último frame."""
        while not self._stop.is_set():
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            try:
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
                else:
                    # tenta reiniciar se leitura falhar
                    self._restart_capture()
            except cv2.error:
                self._restart_capture()

    def get_frame(self):
        """Retorna uma cópia do último frame disponível ou None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _restart_capture(self):
        """Reabre o VideoCapture em caso de falha."""
        try:
            if self._cap:
                self._cap.release()
        except:
            pass
        self._cap = None
        # Pequena pausa antes de reabrir
        time.sleep(0.1)
        # Não recria thread, repete lógica de start sem iniciar nova thread
        try:
            uri = self._camera.create_media_service().GetStreamUri({
                'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                'ProfileToken': self._camera.create_media_service().GetProfiles()[0].token
            }).Uri
            if uri.startswith('rtsp://'):
                uri = uri.replace('rtsp://', f'rtsp://{self.user}:{self.passwd}@')
            self._cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        except:
            pass

    def stop(self) -> None:
        """Encerra captura e libera recursos."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._cap:
            self._cap.release()
        if self._camera:
            try:
                self._camera.devicemgmt.Stop()
            except:
                pass
