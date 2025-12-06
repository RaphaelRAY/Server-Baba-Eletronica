import os
import sys
import time
import cv2
import logging
import json
import base64
import asyncio
import pathlib
from typing import List
from threading import Thread, Event
from queue import Queue, Empty, Full

from fastapi import FastAPI, HTTPException, Response, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is discoverable when executing as a script.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_utils import encode_jpeg

from src.camera import CameraHandler
from src.processing import VideoProcessor
from src.notifications import IdentifiedNotifier, TokenRegistry
from src.monitor.presence_monitor import PresenceMonitor
from src.monitor.position_monitor import PositionMonitor
from src.firebase_setup import FirebaseSetup
from src.db import Database

# Load environment variables from .env
from dotenv import load_dotenv

load_dotenv()

# Configurações e inicialização de câmera e processador
# Logging básico configurável por env (LOG_LEVEL)
level_name = os.getenv("LOG_LEVEL", "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

driver = {
    "host": os.getenv("CAM_HOST", "192.168.0.176"),
    "port": int(os.getenv("CAM_PORT", 80)),
    "user": os.getenv("CAM_USER", "admin"),
    "passwd": os.getenv("CAM_PASS", "123456"),
}

# Fonte de vídeo configurável
video_source = os.getenv("VIDEO_SOURCE", "onvif").lower()
rtsp_url = os.getenv("RTSP_URL")
video_path = os.getenv("VIDEO_PATH")
device_index_env = os.getenv("DEVICE_INDEX")
device_index = int(device_index_env) if device_index_env is not None and device_index_env != "" else None
sync_file_fps = os.getenv("SYNC_FILE_FPS", "true").lower() == "true"
file_fps_env = os.getenv("FILE_FPS")
file_fps = float(file_fps_env) if file_fps_env not in (None, "") else None

# Inicialização de serviços
camera = CameraHandler(
    **driver,
    source=video_source,
    rtsp_url=rtsp_url,
    video_path=video_path,
    device_index=device_index,
    sync_file_fps=sync_file_fps,
    file_fps=file_fps,
)
processor = VideoProcessor(camera)
token_registry = TokenRegistry()
fcm_key = os.getenv("FCM_KEY", "")
notifier_cooldown_env = os.getenv("NOTIFIER_COOLDOWN_SECS")
try:
    notifier_cooldown = int(notifier_cooldown_env) if notifier_cooldown_env else 60
except Exception:
    notifier_cooldown = 60
notifier = IdentifiedNotifier(fcm_key, cooldown=notifier_cooldown)
db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
if db_url:
    # Choose backend by URL scheme
    url_lower = db_url.lower()
    if url_lower.startswith("mongodb://") or url_lower.startswith("mongodb+srv://"):
        database = Database(server=Database.SERVER_MONGO, url=db_url)
    else:
        database = Database(server=Database.SERVER_MYSQL, url=db_url)
else:
    database = Database(server=Database.SERVER_MEMORY)
# Configurable debounce for camera disconnection (in seconds)
cam_disc_secs = float(os.getenv("CAM_DISCONNECT_SECS", "0"))
cam_disc_misses = int(os.getenv("CAM_DISCONNECT_MISSES", "0"))
presence_monitor = PresenceMonitor(
    notifier,
    token_registry,
    database,
    camera_timeout=cam_disc_secs,
    camera_miss_threshold=cam_disc_misses,
)
def _default_posture_model_path() -> str:
    """Pick latest known posture model when env not provided."""
    candidate = pathlib.Path("clsModel2.pt")
    if candidate.is_file():
        return str(candidate)
    return "clsModel.pt"


# Pose detection thresholds (optional configuration)
pose_no_face_frames = int(os.getenv("POSE_NO_FACE_FRAMES", "12"))
posture_model_path = os.getenv("POSTURE_MODEL_PATH") or _default_posture_model_path()
try:
    posture_conf_min = float(
        os.getenv("POSTURE_CONF_MIN", os.getenv("POSE_FACE_CONF_MIN", "0.35"))
    )
except Exception:
    posture_conf_min = 0.35
try:
    posture_stable_frames = int(os.getenv("POSTURE_STABLE_FRAMES", "2"))
except Exception:
    posture_stable_frames = 2
try:
    posture_imgsz = int(os.getenv("POSTURE_IMGSZ", "224"))
except Exception:
    posture_imgsz = 224
try:
    posture_cooldown = float(os.getenv("POSTURE_ANALYSIS_COOLDOWN", "1"))
except Exception:
    posture_cooldown = 1.0
posture_device = os.getenv("POSTURE_DEVICE")
position_monitor = PositionMonitor(
    notifier,
    token_registry,
    database,
    no_face_frames_threshold=pose_no_face_frames,
    model_path=posture_model_path,
    min_confidence=posture_conf_min,
    stable_frames=posture_stable_frames,
    imgsz=posture_imgsz,
    device=posture_device,
    analysis_cooldown_secs=posture_cooldown,
)

# Tolerate missing Firebase setup so app keeps running
try:
    FirebaseSetup().init_firebase(raise_if_missing=False)
    logging.info("Firebase initialized (or using Application Default)")
except Exception as e:
    logging.warning(
        "Firebase not initialized; continuing without push notifications: %s", e
    )

def _get_env_bool(name: str, default: bool) -> bool:
    """Return boolean from env var names like 1/true/yes/on."""
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "y")

# Mostrar poses em janela (configurável via env: SHOW_POSE_WINDOW)
SHOW_POSE_WINDOW = _get_env_bool("SHOW_POSE_WINDOW", True)

# Windows: use Selector event loop to avoid noisy Proactor errors on disconnects
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Simple SSE broker to fan-out new events to connected clients
class SseBroker:
    """Broadcasts events to subscribers via per-client asyncio queues."""

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed: bool = False

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def publish(self, event: dict) -> None:
        # Schedule puts on the app loop from any thread
        if not self._subscribers or self._closed:
            return
        if self._loop is None:
            return
        payload = json.dumps(event, separators=(",", ":"), default=str)
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, payload)
            except Exception:
                # Drop silently if queue is full or loop closed
                pass

    def close(self) -> None:
        """Mark broker closed and wake all subscribers to allow fast shutdown."""
        self._closed = True
        if not self._loop:
            return
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, "__shutdown__")
            except Exception:
                pass


# Global broker instance
sse_broker = SseBroker()

# Eventos para controle de threads de processamento
t_processing_stop = Event()
t_processing_thread = Thread(target=lambda: None)
pose_queue: Queue = Queue(maxsize=2)
detection_queue: Queue = Queue(maxsize=2)
t_pose_thread = Thread(target=lambda: None)
t_detection_thread = Thread(target=lambda: None)


# Função de loop contínuo de processamento
def processing_loop():
    """
    Loop dedicado ao rastreamento automatico PTZ com base na deteccao de pessoa.
    Nao salva nem exibe nada - so move a camera.
    """
    while not t_processing_stop.is_set():
        frame, frame_ts = camera.get_frame_with_timestamp()
        presence_monitor.check_camera(frame)
        if frame is None:
            time.sleep(0.1)
            continue

        _enqueue_frame(pose_queue, frame, frame_ts)
        _enqueue_frame(detection_queue, frame, frame_ts)

        time.sleep(0.01)  # pequena pausa para aliviar CPU


def _enqueue_frame(queue_obj: Queue, frame, frame_ts):
    """Enfileira copia do frame + timestamp, descartando o mais antigo se estiver cheio."""
    try:
        queue_obj.put((frame.copy(), frame_ts), timeout=0.05)
    except Full:
        try:
            queue_obj.get_nowait()
        except Empty:
            pass
        try:
            queue_obj.put((frame.copy(), frame_ts), timeout=0.05)
        except Full:
            pass


def _unwrap_frame_packet(packet):
    """Extrai frame e timestamp de pacotes usados nas filas."""
    if isinstance(packet, (tuple, list)) and len(packet) >= 2:
        return packet[0], packet[1]
    if isinstance(packet, dict):
        return packet.get("frame"), packet.get("timestamp")
    return packet, None

def pose_loop():
    """Processa frames dedicados à análise de postura via classificação."""
    while not t_processing_stop.is_set():
        try:
            packet = pose_queue.get(timeout=0.1)
        except Empty:
            continue

        frame, frame_ts = _unwrap_frame_packet(packet)
        if frame is None:
            continue

        try:
            position_monitor.analyze_frame(
                frame, show=SHOW_POSE_WINDOW, frame_captured_at=frame_ts
            )
        except Exception as exc:
            logging.exception("Erro na thread de pose: %s", exc)


def detection_loop():
    """Processa frames dedicados à detecção e controle PTZ."""
    while not t_processing_stop.is_set():
        try:
            packet = detection_queue.get(timeout=0.1)
        except Empty:
            continue

        try:
            frame, frame_ts = _unwrap_frame_packet(packet)
            if frame is None:
                continue
            results = processor.process_frame_data(frame)
            if results is None:
                results = []
            presence_monitor.handle_detections(results)

            height, width = frame.shape[:2]

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    err_x = (cx - width / 2) / width
                    err_y = (cy - height / 2) / height

                    # Só move se estiver fora da zona morta e PTZ habilitado
                    if camera.ptz_enabled and (abs(err_x) > 0.1 or abs(err_y) > 0.1):
                        camera.control_ptz(err_x, err_y)
                        time.sleep(1.5)

                    break  # só a primeira detecção relevante
        except Exception as exc:
            logging.exception("Erro na thread de detecção: %s", exc)


# FastAPI com contexto de vida (lifespan)
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Iniciando câmera ({video_source}) e loop de análise")
    # Bind loop to SSE broker and hook DB sink
    try:
        sse_broker.set_loop(asyncio.get_running_loop())
        database.set_event_sink(lambda ev: sse_broker.publish(ev))
    except Exception:
        pass
    # 1) Inicia captura de vídeo
    camera.start()
    # 2) Reseta evento e inicia thread de processamento
    t_processing_stop.clear()
    global pose_queue, detection_queue, t_processing_thread, t_pose_thread, t_detection_thread
    pose_queue = Queue(maxsize=2)
    detection_queue = Queue(maxsize=2)
    t_processing_thread = Thread(target=processing_loop, daemon=True)
    t_pose_thread = Thread(target=pose_loop, daemon=True)
    t_detection_thread = Thread(target=detection_loop, daemon=True)
    t_processing_thread.start()
    t_pose_thread.start()
    t_detection_thread.start()


    yield  # aplica as rotas e mantém serviço vivo

    # No shutdown, sinaliza e aguarda thread terminar
    logging.info("Parando loop de análise")
    t_processing_stop.set()
    try:
        sse_broker.close()
    except Exception:
        pass
    t_processing_thread.join(timeout=1)
    t_pose_thread.join(timeout=1)
    t_detection_thread.join(timeout=1)
    camera.stop()
    if SHOW_POSE_WINDOW:
        cv2.destroyAllWindows()


app = FastAPI(lifespan=lifespan)

# CORS configuration
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
# Split by comma, trim spaces; keep ["*"] if wildcard
allowed_origins = (
    ["*"]
    if cors_origins_env.strip() == "*"
    else [o.strip() for o in cors_origins_env.split(",") if o.strip()]
)
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root_status():
    """Retorna status simples da API."""
    return {"status": "connected"}


@app.get("/api/status", response_class=Response)
def status():
    """Verifica se a câmera está conectada."""
    if not camera._cap or not camera._cap.isOpened():
        return Response("Camera is not connected", status_code=503)
    return Response("Camera is connected", status_code=200)


@app.get("/api/snapshot", response_class=Response)
def get_snapshot():
    """Retorna um único frame processado em JPEG."""
    frame = processor.process_frame()
    if frame is None:
        raise HTTPException(503, "Sem frame disponível")
    _, jpg = cv2.imencode(".jpg", frame)
    return Response(jpg.tobytes(), media_type="image/jpeg")


@app.get("/api/pose-snapshot", response_class=Response)
def get_pose_snapshot():
    """Retorna snapshot com overlay da classificação de postura."""
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(503, "Sem frame disponível")

    try:
        results = position_monitor.model(frame)
        if not results:
            raise HTTPException(503, "Sem resultado de pose disponível")

        plotted = results[0].plot()
        jpg_bytes = encode_jpeg(plotted)
        return Response(jpg_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Falha ao gerar pose snapshot: {e}")


@app.get("/api/stream")
def stream():
    """MJPEG stream contínuo sem overlay de texto."""

    # Verifica disponibilidade de frame antes de iniciar o stream
    # Se não houver frame, retorna erro em vez de manter conexão aberta.
    first_frame = camera.get_frame()
    if first_frame is None:
        raise HTTPException(503, "Sem frame disponível para streaming")

    # Define FPS de saída do stream
    target_fps = None
    stream_fps_env = os.getenv("STREAM_FPS")
    if stream_fps_env not in (None, ""):
        try:
            target_fps = float(stream_fps_env)
        except Exception:
            target_fps = None
    if target_fps is None:
        # tenta usar FPS nativo do dispositivo/arquivo se conhecido
        try:
            source_fps = getattr(camera, "_source_fps", None)
            if source_fps:
                target_fps = float(source_fps)
        except Exception:
            target_fps = None
    if target_fps is None:
        # fallback para medir FPS atual em execução
        try:
            fps_stats = camera.get_fps_stats() or {}
            for key in ("current", "mean"):
                candidate = fps_stats.get(key)
                if candidate:
                    target_fps = float(candidate)
                    break
        except Exception:
            target_fps = None
    frame_period = 1.0 / target_fps if target_fps and target_fps > 0 else None

    def mjpeg_generator():
        try:
            while not t_processing_stop.is_set():
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                _, jpg = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                )
                if frame_period:
                    # Ritmo estável do stream
                    time.sleep(frame_period)
        except GeneratorExit:
            # Client disconnected or server shutting down; exit quietly
            return
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected abruptly
            return

    return StreamingResponse(
        mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/events/sse")
async def events_sse():
    """Server-Sent Events stream of newly saved events (no images)."""

    async def event_stream():
        q = sse_broker.subscribe()
        keepalive = 1.0
        try:
            while not t_processing_stop.is_set():
                try:
                    data = await asyncio.wait_for(q.get(), timeout=keepalive)
                    if data == "__shutdown__":
                        break
                    # Proper SSE framing: data: <json>\n\n
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    # Send a comment to keep the connection alive
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            return
        finally:
            sse_broker.unsubscribe(q)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Disable proxy buffering if any (useful for nginx)
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

# Helpers/endpoints para debug de notificacoes
def _parse_tokens(raw) -> list[str]:
    """Normalize token payload (string with commas/newlines or list) into a list."""
    tokens: list[str] = []
    if isinstance(raw, str):
        normalized = raw.replace("\n", ",")
        tokens = [t.strip() for t in normalized.split(",") if t.strip()]
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                tokens.append(item.strip())
    return tokens


DEBUG_NOTIFY_HTML = """
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Debug - Push Notifications</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; max-width: 700px; }
    label { display: block; margin-top: 12px; font-weight: 600; }
    input, textarea, select { width: 100%; padding: 10px; margin-top: 6px; border-radius: 8px; border: 1px solid #1f2937; background: #0b1220; color: #e2e8f0; }
    button { margin-top: 16px; padding: 12px 16px; border: none; border-radius: 10px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .row { display: flex; gap: 12px; margin-top: 12px; }
    .row > div { flex: 1; }
    .status { margin-top: 12px; padding: 10px; border-radius: 8px; background: #0b1220; border: 1px solid #1f2937; }
    .hint { color: #94a3b8; font-size: 0.9rem; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Debug de Notificacoes</h2>
    <div class="hint">Envie push para um token especifico ou para todos os registrados.</div>
    <label for="token">Token (ou deixe vazio e marque "enviar para todos")</label>
    <textarea id="token" rows="3" placeholder="token1, token2 ou um por linha"></textarea>

    <div class="row">
      <div>
        <label for="title">Titulo</label>
        <input id="title" value="[Debug] Ping" />
      </div>
      <div>
        <label for="level">Nivel</label>
        <select id="level">
          <option>Info</option>
          <option>Leve</option>
          <option>Importante</option>
          <option>Urgente</option>
        </select>
      </div>
    </div>

    <label for="message">Mensagem</label>
    <textarea id="message" rows="3" placeholder="Conteudo da notificacao">Teste de debug</textarea>

    <div style="margin-top:12px;">
      <label><input type="checkbox" id="send_all" /> Enviar para todos os tokens registrados</label>
    </div>

    <button id="sendBtn">Enviar notificacao</button>
    <div class="status" id="status">Aguardando envio...</div>
  </div>

  <script>
    const btn = document.getElementById('sendBtn');
    const statusBox = document.getElementById('status');
    const fetchJson = async (url, payload) => {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      return data;
    };
    btn.onclick = async () => {
      btn.disabled = true;
      statusBox.textContent = 'Enviando...';
      try {
        const tokensRaw = document.getElementById('token').value;
        const payload = {
          token: tokensRaw,
          title: document.getElementById('title').value || '[Debug] Ping',
          message: document.getElementById('message').value || 'Teste de debug',
          level: document.getElementById('level').value || 'Info',
          send_all: document.getElementById('send_all').checked
        };
        const res = await fetchJson('/api/debug/notify', payload);
        statusBox.textContent = `Enviado: ${res.sent} notificacoes`;
      } catch (err) {
        statusBox.textContent = 'Erro: ' + err.message;
      } finally {
        btn.disabled = false;
      }
    };
  </script>
</body>
</html>
"""


@app.get("/debug/notify", response_class=HTMLResponse)
def debug_notify_page():
    """Serve uma tela simples para disparar notificacoes de debug via navegador."""
    return DEBUG_NOTIFY_HTML


@app.post("/api/debug/notify")
def debug_notify(data: dict):
    """Enviar notificacao manual para um token ou todos os registrados."""
    title = str(data.get("title") or "[Debug] Ping")
    message = str(data.get("message") or "Teste de debug")
    level = str(data.get("level") or "Info")
    send_all = bool(data.get("all") or data.get("send_all"))

    tokens = _parse_tokens(data.get("tokens") or data.get("token"))
    if send_all or not tokens:
        tokens = token_registry.get_all()
    seen = set()
    tokens = [t for t in tokens if not (t in seen or seen.add(t))]

    if not tokens:
        raise HTTPException(400, "Nenhum token informado ou cadastrado")

    sent = []
    for t in tokens:
        try:
            notifier.send_immediate(t, title=title, message=message, level=level)
            sent.append(t)
        except Exception as exc:
            logging.warning("Falha ao enviar notificacao de debug: %s", exc)
    return {"sent": len(sent), "tokens": sent, "level": level, "title": title}


@app.post("/api/register-token")
def register_token(data: dict):
    """Recebe token FCM e registra para notificações."""
    token = data.get("token")
    if not token:
        raise HTTPException(400, "Token ausente")
    token_registry.add(token)
    return {"status": "ok"}


@app.get("/api/events/image")
def get_event_image(image_path: str | None = Query(None), imagem_path: str | None = Query(None)):
    """Retorna imagem codificada em Base64 usando caminho salvo."""
    path_value = image_path or imagem_path
    if not path_value:
        raise HTTPException(400, "Parametro image_path e obrigatorio")
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Imagem nao encontrada")
    try:
        image_data = path.read_bytes()
    except FileNotFoundError:
        raise HTTPException(404, "Imagem nao encontrada")
    except Exception as exc:
        raise HTTPException(500, f"Erro ao ler imagem: {exc}") from exc
    return {"image_b64": base64.b64encode(image_data).decode("ascii")}


@app.get("/api/events/image/raw")
def get_event_image_raw(image_path: str | None = Query(None), imagem_path: str | None = Query(None)):
    """Retorna imagem como arquivo binario com content-type de imagem."""
    path_value = image_path or imagem_path
    if not path_value:
        raise HTTPException(400, "Parametro image_path e obrigatorio")
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Imagem nao encontrada")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/events")
def get_all_events():
    """Retorna todos os eventos (mais recentes primeiro)."""
    return database.get_all_events()


@app.get("/api/events/noimg")
def get_all_events_noimg():
    """Retorna todos os eventos sem campos de imagem (mais recentes primeiro)."""
    events = database.get_all_events()
    def _strip(ev: dict) -> dict:
        return {k: v for k, v in ev.items() if k not in ("image_b64", "image_bytes")}
    return [_strip(e) for e in events]


@app.get("/api/events/{offset}")
def get_events(offset: int = Path(..., ge=0), limit: int = Query(30, gt=0)):
    """Retorna eventos recentes com campos de imagem (paginado)."""
    return database.get_recent_events(offset=offset, limit=limit)


@app.get("/api/events/{offset}/noimg")
def get_events_noimg(offset: int = Path(..., ge=0), limit: int = Query(30, gt=0)):
    """Retorna eventos recentes sem campos de imagem (paginado)."""
    events = database.get_recent_events(offset=offset, limit=limit)
    def _strip(ev: dict) -> dict:
        return {k: v for k, v in ev.items() if k not in ("image_b64", "image_bytes")}
    return [_strip(e) for e in events]


@app.get("/api/latency")
def get_latency():
    """Retorna JSON com estatísticas de latência."""
    stats = camera.get_latency_stats()
    if not stats:
        raise HTTPException(503, "Ainda não há medições de latência")
    return JSONResponse(
        {
            "last_ms": round(camera.get_last_latency() * 1000, 1),
            "mean_ms": round(stats["mean"] * 1000, 1),
            "min_ms": round(stats["min"] * 1000, 1),
            "max_ms": round(stats["max"] * 1000, 1),
            "samples": stats["count"],
        }
    )


@app.get("/api/performance")
def get_performance_metrics():
    """Retorna métricas de FPS, latência de vídeo e tempo entre detecção e alerta."""
    fps_stats = camera.get_fps_stats() or {}
    latency_stats = camera.get_latency_stats() or {}
    timing_stats = position_monitor.get_timing_stats() or {}

    if not fps_stats and not latency_stats and not timing_stats:
        raise HTTPException(503, "Ainda nao ha metricas coletadas")

    def _round_opt(value, ndigits: int = 2):
        return None if value is None else round(value, ndigits)

    def _ms(value, ndigits: int = 2):
        return None if value is None else round(value * 1000, ndigits)

    fps_payload = (
        {
            "current": _round_opt(fps_stats.get("current")),
            "mean": _round_opt(fps_stats.get("mean")),
            "min": _round_opt(fps_stats.get("min")),
            "max": _round_opt(fps_stats.get("max")),
            "samples": fps_stats.get("samples", 0),
        }
        if fps_stats
        else {}
    )

    latency_payload = (
        {
            "last_ms": _ms(camera.get_last_latency()),
            "mean_ms": _ms(latency_stats.get("mean")),
            "min_ms": _ms(latency_stats.get("min")),
            "max_ms": _ms(latency_stats.get("max")),
            "samples": latency_stats.get("count", 0),
        }
        if latency_stats
        else {}
    )

    timing_payload = {}
    if timing_stats:
        last = timing_stats.get("last") or {}
        averages = timing_stats.get("averages") or {}
        timing_payload = {
            "count": timing_stats.get("count", 0),
            "last": {
                "label": last.get("label"),
                "confidence": last.get("confidence"),
                "process_ms": _ms(last.get("process_s")),
                "alert_ms": _ms(last.get("alert_s")),
                "total_ms": _ms(last.get("total_s")),
            },
            "averages": {
                "process_ms": _ms(averages.get("process_s")),
                "alert_ms": _ms(averages.get("alert_s")),
                "total_ms": _ms(averages.get("total_s")),
            },
        }

    return {
        "fps": fps_payload,
        "latency_ms": latency_payload,
        "detection_to_alert_ms": timing_payload,
    }


if __name__ == "__main__":
    import uvicorn

    # Pass the app object directly to avoid re-import and duplicate logs
    uvicorn.run(app, host="localhost", port=8000, reload=False)
