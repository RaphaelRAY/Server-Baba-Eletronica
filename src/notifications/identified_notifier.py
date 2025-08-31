"""FCM notifier with user identification check and spam prevention."""

import time
from typing import Optional

from .notifier import Notifier


class IdentifiedNotifier:
    """Send FCM notification when a person is identified."""

    def __init__(self, api_key: str, *, threshold: float = 0.9, cooldown: int = 60):
        """Configure thresholds and underlying notifier."""
        self._threshold = threshold
        self._cooldown = cooldown
        self._notifier = Notifier(api_key)
        self._last_sent: float = 0.0

    # Allowed event/notification levels in PT-BR
    _LEVELS = {"Info", "Leve", "Importante", "Urgente"}

    @staticmethod
    def _normalize_level(level: Optional[str]) -> str:
        lvl = (level or "Info").strip().title()
        return lvl if lvl in IdentifiedNotifier._LEVELS else "Info"

    def notify_if_identified(
        self,
        confidence: float,
        token: str,
        *,
        title: str = "Pessoa identificada",
        message: str = "Uma pessoa conhecida foi detectada",
        level: str = "Info",
    ) -> None:
        """Send a notification if confidence exceeds threshold and cooldown expired.

        Adds a level tag to the title (e.g., [Info], [Leve], [Importante], [Urgente]).
        """
        if confidence < self._threshold:
            return

        now = time.time()
        if now - self._last_sent < self._cooldown:
            return

        lvl = self._normalize_level(level)
        prefixed_title = f"[{lvl}] {title}" if lvl else title
        self._notifier.send(token, prefixed_title, message)
        self._last_sent = now

    def notify(self, token: str, *, title: str, message: str, level: str = "Info") -> None:
        """Send a generic notification respecting cooldown with level prefix."""
        now = time.time()
        if now - self._last_sent < self._cooldown:
            return
        lvl = self._normalize_level(level)
        prefixed_title = f"[{lvl}] {title}" if lvl else title
        self._notifier.send(token, prefixed_title, message)
        self._last_sent = now
