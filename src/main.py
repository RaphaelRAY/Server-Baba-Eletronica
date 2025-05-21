# main.py

import os
import io
import cv2
import asyncio
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

# carrega .env
load_dotenv()

# --- Configurações via .env ---
HOST      = os.getenv("HOST", "0.0.0.0")
PORT      = int(os.getenv("PORT", 8000))
RELOAD    = os.getenv("RELOAD", "true").lower() in ("1","true","yes")

CAM_CFG   = {
    "host":   os.getenv("CAM_HOST", "192.168.1.10"),
    "port":   int(os.getenv("CAM_PORT", 80)),
    "user":   os.getenv("CAM_USER", "admin"),
    "passwd": os.getenv("CAM_PASS", "senha")
}
DB_URI    = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/tcc")
FCM_KEY   = os.getenv("FCM_KEY", "")
MODEL_PATH= os.getenv("MODEL_PATH", "yolov11.pt")
DEVICE    = os.getenv("DEVICE", "cpu")

RULES = {
    "baby_class": int(os.getenv("BABY_CLASS", 0)),
    "min_conf":   float(os.getenv("MIN_CONF", 0.5))
}

# --- Importa módulos ---
from src.camera        import CameraHandler
from src.processing    import VideoProcessor
from src.inference     import InferenceEngine
from src.events        import EventManager
from src.db            import Database
from src.notifications import Notifier

# --- Instancia componentes ---
camera    = CameraHandler(**CAM_CFG)
processor = VideoProcessor(camera)
inference = InferenceEngine(model_path=MODEL_PATH, device=DEVICE)
db        = Database(DB_URI)
events    = EventManager(db, RULES)
notifier  = Notifier(api_key=FCM_KEY)

# --- Loop de inferência contínua ---
async def infer_loop():
    camera.start()
    while True:
        frame = processor.get_processed_frame()
        if frame is not None:
            dets = inference.infer(frame)
            evs  = events.analyze(dets)
            if evs:
                events.persist(evs)
                # Para notificações push:
                # for ev in evs:
                #     notifier.send(token, "Alerta", f"{ev['type']} detectado")
        await asyncio.sleep(1.0 / processor.fps)

# --- Lifespan do app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    bg_task = asyncio.create_task(infer_loop())
    yield
    bg_task.cancel()
    with asyncio.suppress(asyncio.CancelledError):
        await bg_task

app = FastAPI(
    title="Electronic Baby Monitor",
    version="0.0.1",
    lifespan=lifespan
)

# --- Endpoints REST ---

@app.get("/api/snapshot", response_class=Response)
def get_snapshot():
    frame = processor.get_processed_frame()
    if frame is None:
        raise HTTPException(503, "Sem frame disponível")
    _, jpg = cv2.imencode('.jpg', frame)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")

@app.get("/api/stream", response_class=StreamingResponse)
def stream():
    def generator():
        while True:
            frame = processor.get_processed_frame()
            if frame is None:
                continue
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/events")
def get_events(limit: int = 50):
    return db.get_recent_events(limit)

# --- Entry point ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD
    )
    