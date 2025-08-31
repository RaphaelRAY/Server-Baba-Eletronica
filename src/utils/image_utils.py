"""Image utilities for encoding frames."""

import cv2


def encode_jpeg(frame, *, quality: int = 85) -> bytes:
    """Encode a BGR frame (numpy array) into JPEG bytes.

    Returns raw JPEG bytes. Raises if encoding fails.
    """
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", frame, params)
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()

