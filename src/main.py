import os
import time
import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
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
    """Retorna um único frame processado (com detecções) em JPEG."""
    frame = processor.process_frame()
    if frame is None:
        raise HTTPException(503, "Sem frame disponível")
    _, jpg = cv2.imencode('.jpg', frame)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")




@app.get("/api/stream")
def stream():
    """MJPEG stream com overlay de latência."""
    def mjpeg_generator():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            # sobrepõe latência atual
            lat = camera.get_last_latency() or 0.0
            cv2.putText(frame, f"Lat: {lat*1000:.1f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            _, jpg = cv2.imencode('.jpg', frame)
            chunk = jpg.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n'
            )

    return StreamingResponse(
        mjpeg_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/api/latency")
def get_latency():
    """
    Retorna JSON com estatísticas de latência:
      - last: última medição em ms
      - mean, min, max: (ms) das últimas 100 leituras
    """
    stats = camera.get_latency_stats()
    if not stats:
        raise HTTPException(503, "Ainda não há medições de latência")
    return JSONResponse({
        "last_ms": round(camera.get_last_latency() * 1000, 1),
        "mean_ms": round(stats["mean"] * 1000, 1),
        "min_ms": round(stats["min"] * 1000, 1),
        "max_ms": round(stats["max"] * 1000, 1),
        "samples": stats["count"]
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="localhost", port=8000, reload=False)
