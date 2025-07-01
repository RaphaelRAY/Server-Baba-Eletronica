# main.py

import os
import cv2
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

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

# --- Importa módulos ---
from src.camera        import CameraHandler
from src.processing    import VideoProcessor

# --- Instancia componentes ---
camera    = CameraHandler(**CAM_CFG)
processor = VideoProcessor(camera)

# --- Inicialização do app ---
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start and stop the camera using FastAPI lifespan events."""
    camera.start()
    try:
        yield
    finally:
        camera.stop()


app = FastAPI(
    title="Electronic Baby Monitor",
    version="0.0.1",
    lifespan=lifespan,
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

# --- Entry point ---
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=RELOAD,
    )
