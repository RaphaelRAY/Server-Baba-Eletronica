import os
import time
import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
import logging

from src.camera import CameraHandler
from src.processing import VideoProcessor

# camera = CameraHandler(
#     host=os.getenv("CAM_HOST", "192.168.0.176"),
#     port=int(os.getenv("CAM_PORT", 80)),
#     user=os.getenv("CAM_USER", "admin"),
#     passwd=os.getenv("CAM_PASS", "123456")
# )
camera = CameraHandler(
    host="192.168.0.176",
    port=80,
    user="admin",
    passwd="123456"
)

processor = VideoProcessor(camera)
app = FastAPI()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize camera and start processing thread."""
    logging.info(" Iniciando câmera ONVIF")
    logging.info(f" Host: {camera.host}, Port: {camera.port}, User: {camera.user}")
    camera.start()
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/api/status", response_class=Response)
def status():
    """Endpoint to check camera status."""
    if not camera._cap or not camera._cap.isOpened():
        return Response(content="Camera is not connected", status_code=503)
    return Response(content="Camera is connected", status_code=200)

    
@app.get("/api/snapshot", response_class=Response)
def get_snapshot():
    """Retorna um único frame processado em JPEG."""
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(503, "Sem frame disponível")
    _, jpg = cv2.imencode('.jpg', frame)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")


def mjpeg_generator():
    """Gera frames JPEG em multipart para MJPEG stream"""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        _, jpg = cv2.imencode('.jpg', frame)
        chunk = jpg.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n'
        )

@app.get("/api/stream")
def stream():
    """Endpoint para streaming contínuo via MJPEG"""
    return StreamingResponse(
        mjpeg_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="localhost", port=8000, reload=False)
