# AGENTS instructions

Este repositório implementa um backend em Python para monitoramento de bebês via FastAPI.

## Execução
- Inicie o servidor com `python src/main.py`.
- Execute os testes com `python -m unittest discover tests`.

## Convenções de código
- Mantenha todos os módulos dentro da pasta `src`.
- Utilize `snake_case` para variáveis e funções.
- Escreva docstrings curtas em português ou inglês.
- Sempre crie testes em `tests/` para novas funcionalidades.

## Commits e Pull Requests
- Use mensagens de commit curtas em inglês.
- Rode a suíte de testes antes de abrir um PR.

## Notificações
- Para enviar push via FCM, use `IdentifiedNotifier` em `src/notifications/identified_notifier.py`.
- Configure a chave de API do FCM via variável de ambiente `FCM_KEY`.
