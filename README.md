# Baba Eletronica Server

Este projeto implementa um backend simples para monitoramento de bebes.
As cameras sao acessadas via ONVIF e os frames podem ser transmitidos
em formato MJPEG via HTTP.

## Testes dos Componentes

Foi adicionada a pasta `tests` com casos de teste para validar partes
isoladas do sistema. Para executar os testes utilize:

```bash
python -m unittest discover tests
```
