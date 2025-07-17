"""Wrapper around Firebase Admin messaging."""

from firebase_admin import messaging


class Notifier:
    """Send push notifications via Firebase Admin SDK."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    def send(self, token: str, title: str, message: str):
        msg = messaging.Message(
            token=token,
            notification=messaging.Notification(title=title, body=message),
        )
        messaging.send(msg)
