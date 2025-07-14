import socket
import os
# Suprime logs do OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
from onvif import ONVIFCamera
from threading import Thread, Event, Lock
from collections import deque
import time

# Timeout global para conexões socket ONVIF (em segundos)
socket.setdefaulttimeout(2)


class CameraHandler:
    """Conecta a uma câmera ONVIF, captura vídeo em thread única e mede latência."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        passwd: str,
        width: int = 640,
        height: int = 480,
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

        # Para medir latência de read()
        self._last_latency: float = None
        self._latencies = deque(maxlen=100)

        # Cache da URI de streaming
        self._stream_uri: str = None

    def start(self) -> None:
        """Inicializa ONVIF (uma vez), abre stream RTSP com timeout e inicia thread."""
        # Se já está rodando, ignora
        if self._cap and self._cap.isOpened() and self._thread and self._thread.is_alive():
            return

        try:
            # 1) Descobre URI apenas na 1ª vez
            if not self._stream_uri:
                self._camera = ONVIFCamera(self.host, self.port, self.user, self.passwd)
                media = self._camera.create_media_service()
                profile = media.GetProfiles()[0]
                uri = media.GetStreamUri({
                    "StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}},
                    "ProfileToken": profile.token
                }).Uri
                if uri.startswith("rtsp://"):
                    uri = uri.replace("rtsp://", f"rtsp://{self.user}:{self.passwd}@")
                self._stream_uri = uri

            # 2) Abre captura com FFmpeg usando a URI cacheada
            #    e configura timeouts de open/read
            self._cap = cv2.VideoCapture(self._stream_uri, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # ↓↓↓ timeouts em milissegundos ↓↓↓
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)

            # -- alternativa usando opts para forçar TCP e stimeout (µs) --
            # opts = ["rtsp_transport","tcp","stimeout","2000000"]
            # self._cap = cv2.VideoCapture(self._stream_uri, cv2.CAP_FFMPEG, opts)

            if not self._cap.isOpened():
                raise RuntimeError(f"Falha ao abrir stream: {self._stream_uri}")

            # 3) Inicia thread de captura
            self._stop.clear()
            self._thread = Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        except Exception as e:
            print(f"[Erro] Falha ao iniciar câmera: {e}")

    def _capture_loop(self):
        """Loop contínuo: lê frame, mede latência e armazena."""
        while not self._stop.is_set():
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            t0 = time.time()
            ret, frame = self._cap.read()
            t1 = time.time()

            if ret:
                latency = t1 - t0
                with self._lock:
                    self._frame = frame
                    self._last_latency = latency
                    self._latencies.append(latency)
            else:
                # reconecta rapidamente usando só a URI
                self._restart_capture()

    def get_frame(self):
        """Retorna uma cópia do último frame ou None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_last_latency(self) -> float:
        """Retorna latência (s) do último read()."""
        with self._lock:
            return self._last_latency

    def get_latency_stats(self) -> dict:
        """
        Estatísticas de latência dos últimos frames:
        { mean, min, max, count } em segundos.
        """
        with self._lock:
            vals = list(self._latencies)
        if not vals:
            return {}
        return {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
        }

    def _restart_capture(self):
        """Reabre o VideoCapture a partir da URI cacheada (sem nova ONVIF)."""
        try:
            if self._cap:
                self._cap.release()
        except:
            pass
        self._cap = None
        time.sleep(0.1)
        if self._stream_uri:
            try:
                self._cap = cv2.VideoCapture(self._stream_uri, cv2.CAP_FFMPEG)
            except:
                pass

    def stop(self) -> None:
        """Para a thread e libera recursos."""
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
