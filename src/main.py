import os
import time
import cv2
import logging
from threading import Thread, Event

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from src.camera import CameraHandler
from src.processing import VideoProcessor

# Configurações e inicialização de câmera e processador
driver = {
    "host": os.getenv("CAM_HOST", "192.168.0.176"),
    "port": int(os.getenv("CAM_PORT", 80)),
    "user": os.getenv("CAM_USER", "admin"),
    "passwd": os.getenv("CAM_PASS", "123456"),
}
camera = CameraHandler(**driver)
processor = VideoProcessor(camera)

# Eventos para controle de threads de processamento
t_processing_stop = Event()
t_processing_thread = Thread(target=lambda: None)


# Função de loop contínuo de processamento
def processing_loop():
    """
    Mantém loop rodando em background: obtém frame, processa e armazena resultado.
    """
    while not t_processing_stop.is_set():
        # Aguarda até 1s pelo próximo frame
        frame = camera.get_frame(wait=True, timeout=1.0)
        if frame is None:
            continue
        # Processa o frame (detecções, anotação, etc.)
        processed = processor.process_frame_data(frame=frame)
        # Aqui você pode salvar ou repassar resultados:
        # ex: publish_to_queue(processed)
        # ou atualizar alguma variável global / cache
        # ...
        time.sleep(0.001)  # pequena pausa para ceder CPU


# FastAPI com contexto de vida (lifespan)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Iniciando câmera ONVIF e loop de análise")
    # 1) Inicia captura de vídeo
    camera.start()
    # 2) Reseta evento e inicia thread de processamento
    t_processing_stop.clear()
    global t_processing_thread
    t_processing_thread = Thread(target=processing_loop, daemon=True)
    t_processing_thread.start()

    yield  # aplica as rotas e mantém serviço vivo

    # No shutdown, sinaliza e aguarda thread terminar
    logging.info("Parando loop de análise")
    t_processing_stop.set()
    t_processing_thread.join(timeout=1)
    camera.stop()


app = FastAPI(lifespan=lifespan)


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
    _, jpg = cv2.imencode('.jpg', frame)
    return Response(jpg.tobytes(), media_type="image/jpeg")


@app.get("/api/stream")
def stream():
    """MJPEG stream com overlay de latência."""
    def mjpeg_generator():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            lat = camera.get_last_latency() or 0.0
            cv2.putText(frame, f"Lat: {lat*1000:.1f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            _, jpg = cv2.imencode('.jpg', frame)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'
            )

    return StreamingResponse(
        mjpeg_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/api/latency")
def get_latency():
    """Retorna JSON com estatísticas de latência."""
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
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
