from __future__ import annotations

import os
from pathlib import Path
import firebase_admin
from firebase_admin import credentials


def _guess_default_path() -> str | None:
    """serviceAccountKey.json ao lado do repo, se existir."""
    root = Path(__file__).resolve().parent.parent
    cand = root / "serviceAccountKey.json"
    return str(cand) if cand.exists() else None


def init_firebase(
    *,
    app_name: str = "[DEFAULT]",
    env_var: str = "FIREBASE_CRED",
    project_id: str | None = None,
    raise_if_missing: bool = True,
):
    """
    Inicializa (ou devolve) uma instância do Firebase Admin SDK.

    - Prioriza o caminho indicado em ``$FIREBASE_CRED``;
    - Se vazio, tenta ``serviceAccountKey.json`` na raiz do projeto;
    - Se ambos faltarem, cai em ADC (GOOGLE_APPLICATION_CREDENTIALS) ou
      lança ``FileNotFoundError`` se ``raise_if_missing`` for True.
    """
    try:
        return firebase_admin.get_app(app_name)
    except ValueError:
        pass  # ainda não existe

    cred_path = os.getenv(env_var) or _guess_default_path()

    # Se não houver JSON, tente Application Default Credentials
    if cred_path:
        cred = credentials.Certificate(cred_path)
    else:
        if raise_if_missing:
            raise FileNotFoundError(
                "Credencial Firebase não encontrada — defina a variável "
                f"{env_var} ou coloque serviceAccountKey.json no projeto."
            )
        cred = credentials.ApplicationDefault()

    opts = {"projectId": project_id} if project_id else None
    return firebase_admin.initialize_app(cred, opts, name=app_name)