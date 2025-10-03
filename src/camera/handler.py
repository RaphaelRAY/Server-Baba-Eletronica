import logging
import math
import os
import re
import socket
import time
from datetime import timedelta
from typing import Any

# Suprime logs do OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
from onvif import ONVIFCamera
from threading import Thread, Event, Lock
from collections import deque

_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    try:
        return float(str(value))
    except Exception:
        return None


def _coerce_timeout_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, timedelta):
        return max(value.total_seconds(), 0.0)
    if isinstance(value, str):
        match = _DURATION_RE.fullmatch(value.strip())
        if not match:
            return None
        days = float(match.group("days") or 0)
        hours = float(match.group("hours") or 0)
        minutes = float(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0)
        return max(days * 86400 + hours * 3600 + minutes * 60 + seconds, 0.0)
    return None

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

        # Configurações de PTZ
        self._ptz_service = None
        self._ptz_profile_token: str | None = None
        self._ptz_configuration_token: str | None = None
        self._ptz_timeout_min: float = 1.0
        self._ptz_timeout_max: float = 10.0
        self._ptz_timeout: float = 1.0
        self._ptz_pan_limit: float | None = None
        self._ptz_tilt_limit: float | None = None
        self._ptz_lock = Lock()
        self._ptz_last_command_ts: float = float("-inf")
        self._ptz_command_interval: float = 0.35
        self._ptz_move_duration: float = 0.4

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
                self._ptz_service = None
                try:
                    self._ptz_service = self._camera.create_ptz_service()
                except Exception as exc:
                    logging.warning("Falha ao criar serviço PTZ: %s", exc)
                    self._ptz_service = None

                profile = self._select_media_profile(media)
                if profile is None:
                    raise RuntimeError("Nenhum profile disponível no serviço de mídia")

                self._setup_ptz_capabilities(profile)

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

    def _select_media_profile(self, media_service) -> Any:
        """Seleciona um profile que possua configuração PTZ quando disponível."""
        try:
            profiles = media_service.GetProfiles()
        except Exception as exc:
            logging.warning("Falha ao obter profiles ONVIF: %s", exc)
            return None
        if not profiles:
            return None
        for profile in profiles:
            if getattr(profile, "PTZConfiguration", None):
                return profile
        return profiles[0]

    def _setup_ptz_capabilities(self, profile: Any) -> None:
        """Guarda tokens, limites de velocidade e timeout compatíveis com a câmera."""
        self._ptz_pan_limit = None
        self._ptz_tilt_limit = None
        self._ptz_timeout_min = 1.0
        self._ptz_timeout_max = 10.0
        self._ptz_timeout = 1.0
        self._ptz_profile_token = getattr(profile, "token", None)
        ptz_config = getattr(profile, "PTZConfiguration", None)
        self._ptz_configuration_token = getattr(ptz_config, "token", None)

        timeout = _coerce_timeout_seconds(
            getattr(ptz_config, "DefaultPTZTimeout", None)
            if ptz_config
            else None
        )
        if timeout is not None:
            self._ptz_timeout = timeout

        # Obtém limites detalhados, se o serviço PTZ estiver disponível
        if not self._ptz_service or not self._ptz_configuration_token:
            # Garante limites padrão seguros
            if self._ptz_pan_limit is None:
                self._ptz_pan_limit = 1.0
            if self._ptz_tilt_limit is None:
                self._ptz_tilt_limit = 1.0
            self._ptz_timeout = self._clamp_timeout(self._ptz_timeout)
            return

        try:
            options = self._ptz_service.GetConfigurationOptions(
                {"ConfigurationToken": self._ptz_configuration_token}
            )
        except Exception as exc:
            logging.warning("Falha ao obter opções PTZ: %s", exc)
            if self._ptz_pan_limit is None:
                self._ptz_pan_limit = 1.0
            if self._ptz_tilt_limit is None:
                self._ptz_tilt_limit = 1.0
            self._ptz_timeout = self._clamp_timeout(self._ptz_timeout)
            return

        spaces = _get_attr_or_key(options, "Spaces")
        pan_limit = self._ptz_pan_limit
        tilt_limit = self._ptz_tilt_limit
        for entry in (
            "ContinuousPanTiltVelocitySpace",
            "PanTiltVelocitySpace",
            "VelocitySpace",
        ):
            for space in _as_iterable(_get_attr_or_key(spaces, entry)):
                x_range = _get_attr_or_key(space, "XRange")
                y_range = _get_attr_or_key(space, "YRange")
                if x_range is None and y_range is None:
                    range_info = _get_attr_or_key(space, "Range")
                    x_range = _get_attr_or_key(range_info, "XRange")
                    y_range = _get_attr_or_key(range_info, "YRange")

                x_min = _coerce_float(_get_attr_or_key(x_range, "Min"))
                x_max = _coerce_float(_get_attr_or_key(x_range, "Max"))
                y_min = _coerce_float(_get_attr_or_key(y_range, "Min"))
                y_max = _coerce_float(_get_attr_or_key(y_range, "Max"))

                if x_min is not None and x_max is not None:
                    limit = max(abs(x_min), abs(x_max))
                    if limit > 0:
                        pan_limit = limit if pan_limit is None else min(pan_limit, limit)
                if y_min is not None and y_max is not None:
                    limit = max(abs(y_min), abs(y_max))
                    if limit > 0:
                        tilt_limit = limit if tilt_limit is None else min(tilt_limit, limit)

        timeout_range = None
        for key in ("TimeoutRange", "PTZTimeout", "Timeout"):
            timeout_range = _get_attr_or_key(options, key)
            if timeout_range:
                break

        min_timeout = _coerce_timeout_seconds(_get_attr_or_key(timeout_range, "Min"))
        max_timeout = _coerce_timeout_seconds(_get_attr_or_key(timeout_range, "Max"))

        if min_timeout is not None and max_timeout is not None and min_timeout <= max_timeout:
            self._ptz_timeout_min = max(0.1, min_timeout)
            self._ptz_timeout_max = max(min_timeout, max_timeout)
        self._ptz_timeout = self._clamp_timeout(self._ptz_timeout)

        self._ptz_pan_limit = pan_limit if pan_limit is not None else 1.0
        self._ptz_tilt_limit = tilt_limit if tilt_limit is not None else 1.0

    def _clamp_timeout(self, value: float | None) -> float:
        base = self._ptz_timeout_min if self._ptz_timeout_min else 0.5
        max_allowed = self._ptz_timeout_max if self._ptz_timeout_max else max(base, 1.0)
        if value is None:
            value = base
        try:
            value = float(value)
        except Exception:
            value = base
        if value <= 0:
            value = base
        value = max(base, value)
        value = min(max_allowed, value)
        return value

    def _refresh_ptz_state(self) -> None:
        if not self.ptz_enabled or self._camera is None:
            return
        try:
            media = self._camera.create_media_service()
        except Exception as exc:
            logging.warning("Falha ao atualizar perfil PTZ: %s", exc)
            return

        if self._ptz_service is None:
            try:
                self._ptz_service = self._camera.create_ptz_service()
            except Exception as exc:
                logging.warning("Falha ao recriar serviço PTZ: %s", exc)
                self._ptz_service = None
                return

        profile = self._select_media_profile(media)
        if profile is None:
            return
        self._setup_ptz_capabilities(profile)

    def _format_timeout(self) -> str:
        seconds = self._clamp_timeout(self._ptz_timeout)
        self._ptz_timeout = seconds
        if abs(seconds - round(seconds)) < 1e-3:
            return f"PT{int(round(seconds))}S"
        numeric = f"{seconds:.2f}".rstrip("0").rstrip(".")
        return f"PT{numeric}S"

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



    def control_ptz(self, err_x: float, err_y: float, kp: float = 1.2) -> None:
        if not self.ptz_enabled or self._camera is None:
            return
        if not self._ptz_profile_token or not self._ptz_service:
            self._refresh_ptz_state()
        if not self._ptz_service or not self._ptz_profile_token:
            return

        with self._ptz_lock:
            now = time.monotonic()
            if now - self._ptz_last_command_ts < self._ptz_command_interval:
                return

            # err_x/err_y já vêm em [-1, 1] do processing_loop

            deadband = 0.02
            if abs(err_x) < deadband:
                err_x = 0.0
            if abs(err_y) < deadband:
                err_y = 0.0

            def shape(e: float) -> float:
                return math.tanh(2.5 * e)

            vx = kp * shape(err_x)
            vy = kp * shape(err_y)

            min_pan, min_tilt = 0.12, 0.12
            if vx != 0.0:
                vx = math.copysign(max(abs(vx), min_pan), vx)
            if vy != 0.0:
                vy = math.copysign(max(abs(vy), min_tilt), vy)

            pan_limit = self._ptz_pan_limit if self._ptz_pan_limit is not None else 1.0
            tilt_limit = self._ptz_tilt_limit if self._ptz_tilt_limit is not None else 1.0
            vx = max(min(vx, pan_limit), -pan_limit)
            vy = max(min(vy, tilt_limit), -tilt_limit)

            if vx == 0.0 and vy == 0.0:
                return

            payload = {
                "ProfileToken": self._ptz_profile_token,
                "Velocity": {"PanTilt": {"x": vx, "y": vy}},
            }

            move_duration = max(0.0, min(self._ptz_move_duration, self._ptz_timeout))

            try:
                self._ptz_service.ContinuousMove(payload)
                if move_duration > 0:
                    self._sleep_interruptible(move_duration)
                self._ptz_service.Stop({"ProfileToken": self._ptz_profile_token})
                self._ptz_last_command_ts = time.monotonic()
            except Exception as exc:
                logging.warning("PTZ move/stop falhou: %s", exc)


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
        self._ptz_service = None
        self._ptz_profile_token = None
        self._ptz_configuration_token = None
