import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.monitor.position_monitor import PositionMonitor
from src.db import Database
from src.db.database import memory_events


class TestPositionMonitor(unittest.TestCase):
    def setUp(self):
        memory_events.clear()

    def _build_monitor(self, **kwargs):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        model = MagicMock()
        model.names = {
            0: "absent",
            1: "left",
            2: "prone",
            3: "sunpine",
            4: "face_covered",
        }
        monitor = PositionMonitor(
            notifier,
            registry,
            Database(server=Database.SERVER_MEMORY),
            model=model,
            analysis_cooldown_secs=0,
            **kwargs,
        )
        return monitor, notifier, registry, model

    @staticmethod
    def _result(idx: int, conf: float = 0.9):
        return SimpleNamespace(probs=SimpleNamespace(top1=idx, top1conf=conf))

    @staticmethod
    def _blank_frame():
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_prone_alert_after_stable_frames(self):
        monitor, notifier, _, model = self._build_monitor(
            stable_frames=2, min_confidence=0.2
        )
        model.return_value = [self._result(2)]

        frame = self._blank_frame()
        monitor.analyze_frame(frame)
        monitor.analyze_frame(frame)

        notifier.notify.assert_called_once()
        call_args = notifier.notify.call_args
        self.assertEqual(call_args.kwargs["title"], "Bebe de brucos")
        self.assertEqual(call_args.kwargs["level"], "Urgente")
        self.assertTrue(monitor.face_down_sent)
        self.assertTrue(any(ev["type"] == "posture_prone" for ev in memory_events))

    def test_side_alert_sets_suspected(self):
        monitor, notifier, _, model = self._build_monitor(
            stable_frames=1, min_confidence=0.2
        )
        model.return_value = [self._result(1)]

        monitor.analyze_frame(self._blank_frame())

        notifier.notify.assert_called_once()
        self.assertEqual(
            notifier.notify.call_args.kwargs["title"], "Posicao suspeita"
        )
        self.assertTrue(monitor.face_down_suspected_sent)

    def test_absent_triggers_after_threshold(self):
        monitor, notifier, _, model = self._build_monitor(
            no_face_frames_threshold=2, min_confidence=0.1
        )
        model.return_value = [self._result(0)]

        frame = self._blank_frame()
        monitor.analyze_frame(frame)
        monitor.analyze_frame(frame)
        monitor.analyze_frame(frame)

        notifier.notify.assert_called_once()
        self.assertEqual(notifier.notify.call_args.kwargs["title"], "Bebe ausente")
        self.assertTrue(
            any(ev["type"] == "posture_absent" for ev in memory_events)
        )

    def test_supine_resets_alerts(self):
        monitor, notifier, _, model = self._build_monitor(
            stable_frames=1, min_confidence=0.1
        )
        model.return_value = [self._result(2)]
        monitor.analyze_frame(self._blank_frame())

        model.return_value = [self._result(3)]
        monitor.analyze_frame(self._blank_frame())

        self.assertFalse(monitor.face_down_sent)
        self.assertFalse(monitor.face_down_suspected_sent)
        notifier.notify.assert_called_once()

    def test_low_confidence_does_not_trigger(self):
        monitor, notifier, _, model = self._build_monitor(
            stable_frames=1, min_confidence=0.8
        )
        model.return_value = [self._result(2, conf=0.5)]

        monitor.analyze_frame(self._blank_frame())

        notifier.notify.assert_not_called()
        self.assertFalse(monitor.face_down_sent)
        self.assertEqual(memory_events, [])

    def test_face_covered_triggers_alert(self):
        monitor, notifier, _, model = self._build_monitor(
            stable_frames=1, min_confidence=0.2
        )
        model.return_value = [self._result(4)]

        monitor.analyze_frame(self._blank_frame())
        model.return_value = [self._result(3)]
        monitor.analyze_frame(self._blank_frame())

        notifier.notify.assert_called_once()
        self.assertEqual(notifier.notify.call_args.kwargs["title"], "Rosto coberto")
        self.assertFalse(monitor.face_covered_sent)
        self.assertTrue(
            any(ev["type"] == "posture_face_covered" for ev in memory_events)
        )

    def test_records_detection_timing(self):
        monitor, notifier, _, model = self._build_monitor(
            stable_frames=1, min_confidence=0.2
        )
        model.return_value = [self._result(2)]

        frame = self._blank_frame()
        with patch("src.monitor.position_monitor.time.perf_counter", side_effect=[1.05, 1.06]):
            monitor.analyze_frame(frame, frame_captured_at=1.0)

        stats = monitor.get_timing_stats()
        self.assertEqual(stats["count"], 1)
        last = stats["last"]
        self.assertAlmostEqual(last["process_s"], 0.05)
        self.assertAlmostEqual(last["total_s"], 0.06)


if __name__ == "__main__":
    unittest.main()
