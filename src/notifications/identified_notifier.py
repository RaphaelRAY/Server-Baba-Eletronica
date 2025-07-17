"""FCM notifier with user identification check and spam prevention."""

import time
from typing import Optional

from .notifier import Notifier


class IdentifiedNotifier:
    """Send FCM notification when a person is identified."""

    def __init__(self, api_key: str, *, threshold: float = 0.9, cooldown: int = 60):
        self._threshold = threshold
        self._cooldown = cooldown
        self._notifier = Notifier(api_key)
        self._last_sent: float = 0.0

    def notify_if_identified(self, confidence: float, token: str, *, title: str = "Pessoa identificada", message: str = "Uma pessoa conhecida foi detectada") -> None:
        """Send a notification if confidence exceeds threshold and cooldown expired."""
        if confidence < self._threshold:
            return

        now = time.time()
        if now - self._last_sent < self._cooldown:
            return

        self._notifier.send(token, title, message)
        self._last_sent = now

    def notify(self, token: str, *, title: str, message: str) -> None:
        """Send a generic notification respecting cooldown."""
        now = time.time()
        if now - self._last_sent < self._cooldown:
            return
        self._notifier.send(token, title, message)
        self._last_sent = now
