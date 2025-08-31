import os
import shutil
import unittest
import base64

from src.db.database import memory_events
from src.db import Database


class TestEventImages(unittest.TestCase):
    def setUp(self):
        memory_events.clear()
        # Use a temporary directory under tests
        self.tmp_dir = os.path.join("tests", "_tmp_events")
        os.makedirs(self.tmp_dir, exist_ok=True)
        # Point EVENTS_DIR to temp
        os.environ["EVENTS_DIR"] = self.tmp_dir

    def tearDown(self):
        # Cleanup temp directory
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass
        os.environ.pop("EVENTS_DIR", None)

    def test_save_event_with_image_bytes(self):
        db = Database(server=Database.SERVER_MEMORY)
        img_bytes = b"\xff\xd8\xff\xdbFAKEJPEGDATA\xff\xd9"  # minimal bytes
        db.save_event({
            "type": "test_img",
            "confidence": 0.7,
            "level": "Info",
            "image_bytes": img_bytes,
        })

        events = db.get_recent_events()
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["type"], "test_img")
        self.assertIn("image_path", ev)
        self.assertTrue(os.path.isfile(ev["image_path"]))
        with open(ev["image_path"], "rb") as fh:
            self.assertEqual(fh.read(), img_bytes)
        self.assertIn("image_b64", ev)
        decoded = base64.b64decode(ev["image_b64"]) if ev["image_b64"] else b""
        self.assertEqual(decoded, img_bytes)


if __name__ == "__main__":
    unittest.main()

