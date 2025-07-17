import os

import firebase_admin
from firebase_admin import credentials


def init_firebase() -> None:
    """Initialize Firebase app using FIREBASE_CRED or local JSON."""
    if firebase_admin._apps:
        return

    path = os.getenv("FIREBASE_CRED")
    if not path:
        root = os.path.dirname(os.path.dirname(__file__))
        default = os.path.join(root, "serviceAccountKey.json")
        if os.path.exists(default):
            path = default

    if not path or not os.path.exists(path):
        return

    cred = credentials.Certificate(path)
    firebase_admin.initialize_app(cred)
