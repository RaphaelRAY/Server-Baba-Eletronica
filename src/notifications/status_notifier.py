import time
from .notifier import Notifier


class StatusNotifier:
    """Send alerts for no detections or camera disconnects."""

    def __init__(self, api_key: str, *, cooldown: int = 60):
        self._notifier = Notifier(api_key)
        self._cooldown = cooldown
        self._last_no_person = -cooldown
        self._last_disconnect = -cooldown

    def notify_no_person(self, token: str,
                         *, title: str = "Sem detec\u00e7\u00e3o",
                         message: str = "Nenhuma pessoa detectada") -> None:
        now = time.time()
        if now - self._last_no_person < self._cooldown:
            return
        self._notifier.send(token, title, message)
        self._last_no_person = now

    def notify_disconnect(self, token: str,
                          *, title: str = "C\u00e2mera desconectada",
                          message: str = "A c\u00e2mera parou de enviar imagens") -> None:
        now = time.time()
        if now - self._last_disconnect < self._cooldown:
            return
        self._notifier.send(token, title, message)
        self._last_disconnect = now


