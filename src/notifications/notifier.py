"""Wrapper around Firebase Admin messaging."""

import logging
import os

try:
    from firebase_admin import messaging  # type: ignore
except Exception:  # firebase not available/initialized
    messaging = None  # type: ignore


def _env_flag(name: str, default: bool = False) -> bool:
    """Return True when the given environment flag is truthy."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Notifier:
    """Send push notifications via Firebase Admin SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        log_details: bool | None = None,
    ) -> None:
        """Store API key and configure detailed logging via FCM_LOG_DETAILS."""
        self._api_key = api_key
        self._log_details = (
            _env_flag("FCM_LOG_DETAILS") if log_details is None else log_details
        )

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
            if self._log_details:
                logging.info(
                    "Sending FCM message to token %s with title %r and body %r",
                    token,
                    title,
                    message,
                )
            response = messaging.send(msg)
            if self._log_details:
                logging.info(
                    "FCM message delivered to token %s; response id: %s",
                    token,
                    response,
                )
        except Exception as exc:
            logging.warning(
                "Failed to send FCM message to token %s; continuing without crash: %s",
                token,
                exc,
            )
