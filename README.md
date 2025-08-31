# Baba Eletronica Server

Este projeto implementa um backend simples para monitoramento de bebês.
As cameras sao acessadas via ONVIF e os frames podem ser transmitidos
em formato MJPEG via HTTP.

## Executar

Para iniciar o servidor e acessar a câmera, rode:

```bash
python -m src.main
```

O snapshot pode ser obtido em `/api/snapshot` e o streaming em `/api/stream`.

A aplicação usa eventos de lifespan do FastAPI para ligar e desligar a câmera
automaticamente.

## Eventos com imagem

- Ao detectar eventos (ex.: ausência, câmera desconectada, bebê de bruços), o servidor salva um snapshot JPEG no diretório configurado por `EVENTS_DIR` (padrão: `data/events`).
- As rotas `/api/events` e `/api/events/{offset}` retornam, além de `type`, `confidence` e `timestamp`, os campos opcionais `image_path` e `image_b64` (base64 do snapshot) quando disponíveis.

Para configurar o diretório de snapshots, adicione ao `.env`:

```
EVENTS_DIR=data/events
```

## Testes dos Componentes

Foi adicionada a pasta `tests` com casos de teste para validar partes
isoladas do sistema. Para executar os testes utilize:

```bash
python -m unittest discover tests
```
