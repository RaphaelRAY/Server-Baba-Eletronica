"""Wrapper around Firebase Admin messaging."""

from firebase_admin import messaging


class Notifier:
    """Send push notifications via Firebase Admin SDK."""

    def __init__(self, api_key: str | None = None):
        """Store API key if needed for external services."""
        self._api_key = api_key

    def send(self, token: str, title: str, message: str):
        """Send a push notification to a single token."""
        msg = messaging.Message(
            token=token,
            notification=messaging.Notification(title=title, body=message),
        )
        messaging.send(msg)
