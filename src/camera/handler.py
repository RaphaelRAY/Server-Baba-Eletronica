import socket
import os
import logging
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
    """Captura vídeo (ONVIF/RTSP/arquivo/dispositivo) em thread e mede latência."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        passwd: str,
        width: int = 640,
        height: int = 480,
        *,
        source: str = "onvif",
        rtsp_url: str | None = None,
        video_path: str | None = None,
        device_index: int | None = None,
        sync_file_fps: bool = True,
        file_fps: float | None = None,
    ):
        """Configura fonte de vídeo e opções.

        - source: "onvif" | "rtsp" | "file" | "device"
        - rtsp_url: URL completa quando source == "rtsp"
        - video_path: caminho para arquivo quando source == "file"
        - device_index: índice do dispositivo quando source == "device"
        """
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.width = width
        self.height = height

        self.source = (source or "onvif").lower()
        self.rtsp_url = rtsp_url
        self.video_path = video_path
        try:
            self.device_index = int(device_index) if device_index is not None else None
        except Exception:
            self.device_index = None

        # Habilita PTZ apenas para ONVIF
        self.ptz_enabled = self.source == "onvif"

        # Controle de FPS (somente para source=file por padrão)
        self.sync_file_fps = bool(sync_file_fps)
        try:
            self.file_fps = float(file_fps) if file_fps else None
        except Exception:
            self.file_fps = None

        self._source_fps: float | None = None
        self._frame_period: float | None = None
        self._next_frame_ts: float | None = None

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

        # Reconexão
        self._reconnecting: bool = False
        self._reconnect_thread: Thread | None = None
        self._reconnect_delay: float = 5.0

    def _sleep_interruptible(self, seconds: float, step: float = 0.1) -> None:
        """Sleep in small steps so stop() can interrupt long waits."""
        deadline = time.time() + max(0.0, seconds)
        while not self._stop.is_set() and time.time() < deadline:
            time.sleep(min(step, max(0.0, deadline - time.time())))

    def start(self) -> None:
        """Inicializa ONVIF (uma vez), abre stream RTSP com timeout e inicia thread."""
        # Se já está rodando, ignora
        if self._cap and self._cap.isOpened() and self._thread and self._thread.is_alive():
            return

        try:
            self._open_connections()

        except Exception as e:
            logging.error("Falha ao iniciar câmera: %s", e)
            # agenda reconexão periódica
            self._schedule_reconnect()

    def _capture_loop(self):
        """Loop contínuo: lê frame, mede latência e armazena."""
        while not self._stop.is_set():
            if not self._cap or not self._cap.isOpened():
                # tenta reabrir com atraso para evitar loop apertado
                self._restart_capture(delay=self._reconnect_delay)
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

                # Ritmo de saída: manter FPS do arquivo, se configurado
                if (
                    self.source == "file"
                    and self._frame_period
                    and self.sync_file_fps
                ):
                    if self._next_frame_ts is None:
                        self._next_frame_ts = time.time() + self._frame_period
                    else:
                        self._next_frame_ts += self._frame_period
                    remaining = self._next_frame_ts - time.time()
                    if remaining > 0:
                        self._sleep_interruptible(remaining)
            else:
                # reconecta com atraso fixo de 5s
                self._restart_capture(delay=self._reconnect_delay)

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

    def _restart_capture(self, *, delay: float = 5.0):
        """Reabre o VideoCapture a partir da URI cacheada após um atraso."""
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        # Wait but allow stop() to interrupt
        self._sleep_interruptible(delay)
        # Reabra conforme a fonte
        try:
            if self.source == "device":
                index = 0 if self.device_index is None else int(self.device_index)
                self._cap = cv2.VideoCapture(index)
            else:
                if not self._stream_uri:
                    return
                self._cap = cv2.VideoCapture(self._stream_uri, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
            # Recalcula FPS e período para arquivo
            self._source_fps = None
            self._frame_period = None
            self._next_frame_ts = None
            if self.source == "file":
                try:
                    fps_val = float(self._cap.get(cv2.CAP_PROP_FPS) or 0)
                    if fps_val and fps_val > 0:
                        self._source_fps = fps_val
                except Exception:
                    pass
                if self.file_fps and self.file_fps > 0:
                    self._source_fps = self.file_fps
                if self._source_fps and self._source_fps > 0:
                    self._frame_period = 1.0 / self._source_fps
        except Exception as e:
            logging.warning("Falha ao reabrir stream: %s", e)

    def _open_connections(self) -> None:
        """Prepara a fonte de vídeo escolhida e inicia captura + thread."""
        # 1) Determina a URI/fonte de captura conforme source
        if not self._stream_uri:
            if self.source == "onvif":
                self._camera = ONVIFCamera(self.host, self.port, self.user, self.passwd)
                media = self._camera.create_media_service()
                profile = media.GetProfiles()[0]
                uri = media.GetStreamUri({
                    "StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}},
                    "ProfileToken": profile.token,
                }).Uri
                if uri.startswith("rtsp://"):
                    uri = uri.replace("rtsp://", f"rtsp://{self.user}:{self.passwd}@")
                self._stream_uri = uri
            elif self.source == "rtsp":
                if not self.rtsp_url:
                    raise ValueError("RTSP_URL não informado para source=rtsp")
                self._stream_uri = self.rtsp_url
            elif self.source == "file":
                if not self.video_path:
                    raise ValueError("VIDEO_PATH não informado para source=file")
                self._stream_uri = self.video_path
            elif self.source == "device":
                # Para device não usamos URI string, abrimos com índice
                self._stream_uri = None
            else:
                raise ValueError(f"Fonte de vídeo inválida: {self.source}")

        # 2) Abre captura com backend apropriado
        if self.source == "device":
            # Tenta abrir dispositivo por índice
            index = 0 if self.device_index is None else int(self.device_index)
            # Evita passar CAP_FFMPEG para dispositivos
            self._cap = cv2.VideoCapture(index)
        else:
            self._cap = cv2.VideoCapture(self._stream_uri, cv2.CAP_FFMPEG)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)

        # FPS de origem (para arquivos geralmente disponível)
        self._source_fps = None
        self._frame_period = None
        self._next_frame_ts = None
        try:
            fps_val = float(self._cap.get(cv2.CAP_PROP_FPS) or 0)
            if fps_val and fps_val > 0:
                self._source_fps = fps_val
        except Exception:
            pass
        # Override opcional via parâmetro (útil quando CAP_PROP_FPS = 0)
        if self.source == "file":
            if self.file_fps and self.file_fps > 0:
                self._source_fps = self.file_fps
            # Define período caso tenhamos FPS
            if self._source_fps and self._source_fps > 0:
                self._frame_period = 1.0 / self._source_fps

        if not self._cap.isOpened():
            ident = (
                f"device index {self.device_index}"
                if self.source == "device"
                else self._stream_uri
            )
            raise RuntimeError(f"Falha ao abrir stream: {ident}")

        # 3) Inicia thread de captura
        self._stop.clear()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _schedule_reconnect(self) -> None:
        """Agenda tentativas de reconexão a cada 5s até sucesso."""
        if self._reconnecting or self._stop.is_set():
            return

        def _loop():
            self._reconnecting = True
            while not self._stop.is_set():
                try:
                    # Allow early exit on stop
                    self._sleep_interruptible(self._reconnect_delay)
                    self._open_connections()
                    logging.info("Câmera reconectada com sucesso")
                    break
                except Exception as e:
                    logging.warning("Tentativa de reconexão falhou: %s", e)
                    continue
            self._reconnecting = False

        self._reconnect_thread = Thread(target=_loop, daemon=True)
        self._reconnect_thread.start()

    def is_running(self) -> bool:
        """Retorna True se a thread e a captura estiverem ativas."""
        return bool(self._cap and self._cap.isOpened() and self._thread and self._thread.is_alive())

    def control_ptz(self, err_x: float, err_y: float, kp: float = 0.6):
        """Controla PTZ (ativo somente em ONVIF)."""
        if not self.ptz_enabled or self._camera is None:
            return
        try:
            ptz = self._camera.create_ptz_service()
            media = self._camera.create_media_service()
            profile = media.GetProfiles()[0]
            token = profile.token
        
            # Aplica ganho proporcional
            vx = kp * err_x
            vy = kp * err_y
        
            # Limita velocidades entre -1.0 e 1.0
            vx = max(min(vx, 1.0), -1.0)
            vy = max(min(vy, 1.0), -1.0)
        
            # Monta comando de movimento
            ptz.ContinuousMove({
                "ProfileToken": token,
                "Velocity": {
                    "PanTilt": {
                        "x": -vx,
                        "y": -vy  # Inverte se necessário (ajuste depende da câmera)
                    }
                }
            })
        
            # Aguarda movimento curto, depois para
            time.sleep(0.2)
            ptz.Stop({"ProfileToken": token})
        
        except Exception as e:
            logging.warning("Falha no controle PTZ: %s", e)


    def stop(self) -> None:
        """Para a thread e libera recursos."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        # Ensure reconnection loop ends as well
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            # Give it a short chance to exit
            self._reconnect_thread.join(timeout=1)
        if self._cap:
            self._cap.release()
        if self._camera:
            try:
                self._camera.devicemgmt.Stop()
            except:
                pass
