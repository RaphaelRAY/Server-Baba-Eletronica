# Baba Eletronica Server

Este projeto implementa um backend simples para monitoramento de bebes.
As cameras sao acessadas via ONVIF e os frames podem ser transmitidos
em formato MJPEG via HTTP.

## Executar

Para iniciar o servidor e acessar a câmera, rode:

```bash
python src/main.py
```

O snapshot pode ser obtido em `/api/snapshot` e o streaming em `/api/stream`.

## Testes dos Componentes

Foi adicionada a pasta `tests` com casos de teste para validar partes
isoladas do sistema. Para executar os testes utilize:

```bash
python -m unittest discover tests
```
