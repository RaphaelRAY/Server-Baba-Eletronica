from pyfcm import FCMNotification


class Notifier:
    """Send push notifications via FCM."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = FCMNotification(api_key=api_key)

    def send(self, token: str, title: str, message: str):
        self._client.notify_single_device(
            registration_id=token, message_title=title, message_body=message
        )
