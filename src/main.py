import os
import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

from src.camera import CameraHandler
from src.processing import VideoProcessor

camera = CameraHandler(
    host=os.getenv("CAM_HOST", "192.168.1.10"),
    port=int(os.getenv("CAM_PORT", 80)),
    user=os.getenv("CAM_USER", "admin"),
    passwd=os.getenv("CAM_PASS", "senha")
)
processor = VideoProcessor(camera)
app = FastAPI()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize camera and start processing thread."""
    camera.start()
    #processor.start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/api/snapshot", response_class=Response)
def get_snapshot():
    frame = processor.get_processed_frame()
    if frame is None:
        raise HTTPException(503, "Sem frame disponível")
    _, jpg = cv2.imencode('.jpg', frame)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
