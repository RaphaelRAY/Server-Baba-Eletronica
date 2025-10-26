"""Wrapper around Firebase Admin messaging."""

import logging
import os

try:
    import firebase_admin  # type: ignore
except Exception:  # firebase-admin not installed
    firebase_admin = None  # type: ignore

try:
    from firebase_admin import messaging  # type: ignore
except Exception:  # firebase not available/initialized
    messaging = None  # type: ignore

try:
    from src.firebase_setup import FirebaseSetup
except Exception:  # defer initialization if module unavailable
    FirebaseSetup = None  # type: ignore


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

        if not self._ensure_firebase_initialized():
            logging.warning(
                "Firebase app not initialized; dropping notification to token %s",
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

    @staticmethod
    def _ensure_firebase_initialized() -> bool:
        """Ensure a Firebase app exists before sending messages."""
        if firebase_admin is None:
            return False

        try:
            firebase_admin.get_app()  # type: ignore[attr-defined]
            return True
        except ValueError:
            pass  # No default app yet.
        except Exception as exc:  # pragma: no cover - unexpected failure
            logging.debug("Unexpected Firebase get_app failure: %s", exc, exc_info=True)
            return False

        if FirebaseSetup is None:
            logging.debug("FirebaseSetup helper unavailable; cannot initialize Firebase")
            return False

        try:
            FirebaseSetup().init_firebase(raise_if_missing=False)
            return True
        except Exception as exc:
            logging.warning("Firebase initialization failed: %s", exc)
            return False
