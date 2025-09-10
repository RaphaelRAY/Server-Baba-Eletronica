# Baba Eletrônica Server

Backend em FastAPI para monitoramento de bebês com captura de vídeo (ONVIF/RTSP/arquivo/dispositivo), análise com YOLO e notificação via FCM.

## Preparação

- Requisitos: Python 3.10+ (recomendado), FFmpeg disponível no sistema para RTSP.
- Crie e ative um ambiente virtual e instale dependências:
  - `python -m venv .venv`
  - `source .venv/bin/activate` (Linux/macOS) ou `.venv\Scripts\activate` (Windows)
  - `pip install -r requirements.txt`

## Configuração (.env)

Copie `.env.example` para `.env` e ajuste conforme a fonte de vídeo e integrações:

- Fonte de vídeo (`VIDEO_SOURCE=onvif|rtsp|file|device`)
  - ONVIF: `CAM_HOST`, `CAM_PORT`, `CAM_USER`, `CAM_PASS`
  - RTSP: `RTSP_URL=rtsp://user:pass@ip:554/stream`
  - Arquivo: `VIDEO_PATH=/caminho/video.mp4` e opcional `SYNC_FILE_FPS=true`, `FILE_FPS=30`
  - Dispositivo: `DEVICE_INDEX=0`
- Stream MJPEG: `STREAM_FPS=15` (opcional, usa FPS do arquivo se não definido)
- Notificações (FCM): `FCM_KEY` e `NOTIFIER_COOLDOWN_SECS=60`
- Deduplicação de eventos: `EVENT_COOLDOWN_SECS=60` e overrides por tipo `EVENT_COOLDOWN_face_down=60`
- Diretório de imagens de eventos: `EVENTS_DIR=data/events`
- CORS (frontend): `CORS_ORIGINS=http://localhost:3000`
- Log: `LOG_LEVEL=INFO`

## Executar

Inicie o servidor:

```bash
python src/main.py
```

Por padrão o servidor sobe em `http://localhost:8000`.

## Endpoints

- GET `/api/status`: estado da câmera (200 conectado, 503 desconectado).
- GET `/api/snapshot`: snapshot JPEG do frame processado (503 se indisponível).
- GET `/api/pose-snapshot`: snapshot JPEG com overlay de pose (503/500 on error).
- GET `/api/stream`: stream MJPEG contínuo com overlay de latência.
- POST `/api/register-token` {"token": "..."}: registra token FCM.
- GET `/api/events`: lista todos os eventos (mais recentes primeiro).
- GET `/api/events/{offset}` (query `limit`): eventos paginados sem imagem embutida.
- GET `/api/latency`: estatísticas de latência (ms) do capture.

## Eventos e imagens

- Em eventos (ausência, câmera desconectada/conectada, bebê de bruços/suspeito), um snapshot pode ser salvo em `EVENTS_DIR`.
- As rotas de eventos retornam também `image_path` e, quando aplicável, `image_b64`.

## Testes

Execute a suíte de testes:

```bash
python -m unittest discover tests
```
