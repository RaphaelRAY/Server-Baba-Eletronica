"""Wrapper around Firebase Admin messaging."""

import logging

try:
    from firebase_admin import messaging  # type: ignore
except Exception:  # firebase not available/initialized
    messaging = None  # type: ignore


class Notifier:
    """Send push notifications via Firebase Admin SDK."""

    def __init__(self, api_key: str | None = None):
        """Store API key if needed for external services."""
        self._api_key = api_key

    def send(self, token: str, title: str, message: str):
        """Send a push notification to a single token.

        If Firebase is not initialized or sending fails, log and continue.
        """
        if messaging is None:
            logging.warning(
                "Firebase messaging unavailable; dropping notification to token %s",
                token,
            )
            return

        try:
            msg = messaging.Message(
                token=token,
                notification=messaging.Notification(title=title, body=message),
            )
            messaging.send(msg)
        except Exception as exc:
            logging.warning(
                "Failed to send FCM message; continuing without crash: %s", exc
            )
