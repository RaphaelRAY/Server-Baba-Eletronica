import unittest
from unittest.mock import MagicMock, patch, call

from src.monitor.position_monitor import PositionMonitor, LEFT_COLOR, RIGHT_COLOR


class TestPositionMonitor(unittest.TestCase):
    def test_face_down_detection(self):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        monitor = PositionMonitor(notifier, registry, model=MagicMock(), face_down_margin=0)

        face_down = MagicMock()
        kps = MagicMock()
        # nose_y > shoulders_y triggers notification
        kps.xy = [
            (0, 50),  # nose
            (0, 0),   # placeholders for remaining keypoints
            (0, 0),
            (0, 0),
            (0, 0),
            (-40, 20),  # left_shoulder
            (40, 20),  # right_shoulder
        ]
        face_down.keypoints = kps

        safe = MagicMock()
        kps_safe = MagicMock()
        kps_safe.xy = [
            (0, 10),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (-40, 20),
            (40, 20),
        ]
        safe.keypoints = kps_safe

        monitor.handle_pose([face_down])
        notifier.notify.assert_called_once()

        monitor.handle_pose([safe])
        monitor.handle_pose([face_down])
        self.assertEqual(notifier.notify.call_count, 2)

    @patch("src.monitor.position_monitor.cv2")
    def test_pose_window_resizable(self, cv2_mock):
        cv2_mock.WINDOW_NORMAL = 1
        cv2_mock.WINDOW_GUI_EXPANDED = 2
        cv2_mock.LINE_AA = 16
        cv2_mock.WND_PROP_VISIBLE = 1
        cv2_mock.getWindowProperty.return_value = 1
        cv2_mock.cvtColor.side_effect = lambda img, code: img
        channel = MagicMock()
        cv2_mock.split.return_value = (channel, channel, channel)
        cv2_mock.equalizeHist.return_value = channel
        cv2_mock.merge.return_value = MagicMock()

        notifier = MagicMock()
        registry = MagicMock()
        result = MagicMock()
        points = [
            (10.0, 10.0),  # nose
            (11.0, 10.5),
            (9.0, 10.5),
            (11.0, 11.0),
            (9.0, 11.0),
            (6.0, 20.0),   # left_shoulder
            (16.0, 20.0),  # right_shoulder
            (4.0, 25.0),   # left_elbow
            (18.0, 25.0),  # right_elbow
            (2.0, 30.0),   # left_wrist
            (20.0, 30.0),  # right_wrist
            (6.0, 35.0),   # left_hip
            (16.0, 35.0),  # right_hip
            (5.0, 45.0),   # left_knee
            (17.0, 45.0),  # right_knee
            (4.0, 55.0),   # left_ankle
            (18.0, 55.0),  # right_ankle
        ]
        confs = [0.9] * len(points)
        result.keypoints = type("KP", (), {"xy": points, "conf": confs})()
        result.plot.return_value = MagicMock()
        model = MagicMock()
        model.predict.return_value = [result]
        monitor = PositionMonitor(notifier, registry, model=model)

        frame = MagicMock()
        frame_copy = MagicMock(name="frame_copy")
        frame.copy.return_value = frame_copy

        monitor.analyze_frame(frame, show=True)
        monitor.analyze_frame(frame, show=True)

        expected_flag = cv2_mock.WINDOW_NORMAL | cv2_mock.WINDOW_GUI_EXPANDED
        cv2_mock.namedWindow.assert_called_once_with("Pose", expected_flag)
        self.assertEqual(cv2_mock.imshow.call_count, 2)
        for args, kwargs in cv2_mock.imshow.call_args_list:
            self.assertEqual(args[0], "Pose")
            self.assertIs(args[1], frame_copy)
        self.assertEqual(cv2_mock.waitKey.call_count, 2)
        self.assertTrue(any(call_kwargs.args[3] == LEFT_COLOR for call_kwargs in cv2_mock.circle.call_args_list))
        self.assertTrue(any(call_kwargs.args[3] == RIGHT_COLOR for call_kwargs in cv2_mock.circle.call_args_list))
        result.plot.assert_not_called()

    @patch("src.monitor.position_monitor.cv2")
    def test_pose_window_updates_without_results(self, cv2_mock):
        cv2_mock.WINDOW_NORMAL = 1
        cv2_mock.WINDOW_GUI_EXPANDED = 2
        cv2_mock.LINE_AA = 16
        cv2_mock.cvtColor.side_effect = lambda img, code: img
        channel = MagicMock()
        cv2_mock.split.return_value = (channel, channel, channel)
        cv2_mock.equalizeHist.return_value = channel
        cv2_mock.merge.return_value = MagicMock()

        notifier = MagicMock()
        registry = MagicMock()
        model = MagicMock()
        model.predict.return_value = []
        monitor = PositionMonitor(notifier, registry, model=model)

        frame = MagicMock()
        frame_copy = MagicMock(name="frame_copy")
        frame.copy.return_value = frame_copy
        monitor.analyze_frame(frame, show=True)

        cv2_mock.namedWindow.assert_called_once()
        cv2_mock.imshow.assert_called_once_with("Pose", frame_copy)
        cv2_mock.waitKey.assert_called_once_with(1)
        cv2_mock.circle.assert_not_called()

    @patch("src.monitor.position_monitor.YOLO")
    def test_lazy_model_loading(self, yolo_mock):
        notifier = MagicMock()
        registry = MagicMock()
        monitor = PositionMonitor(notifier, registry, model=None)

        yolo_mock.assert_not_called()
        loaded = monitor._get_model()
        yolo_mock.assert_called_once_with("yolo11x-pose.pt")
        self.assertIs(loaded, yolo_mock.return_value)

    def test_face_visibility_threshold(self):
        notifier = MagicMock()
        registry = MagicMock()
        monitor = PositionMonitor(notifier, registry, model=MagicMock(), face_conf_min=0.5)

        confs_visible = [0.1, 0.2, 0.6, None, 0.1]
        confs_hidden = [0.1, 0.2, 0.3, None, 0.1]

        self.assertTrue(monitor._is_face_visible(confs_visible))
        self.assertFalse(monitor._is_face_visible(confs_hidden))

    def test_face_down_detection_helper(self):
        from types import SimpleNamespace

        notifier = MagicMock()
        registry = MagicMock()
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            face_down_margin=5.0,
            risk_threshold=0.7,
        )

        points = [
            (0, 50),  # nose
            (0, 48),
            (0, 48),
            (0, 47),
            (0, 47),
            (-40, 20),
            (40, 20),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (-30, 40),
            (30, 40),
        ]
        confs = [0.9, 0.8, 0.75, 0.7, 0.65, 0.9, 0.9] + [0.8] * 6
        keypoints = SimpleNamespace(xy=points, conf=confs)

        metrics = monitor._compute_pose_metrics(points, confs, keypoints)
        self.assertTrue(metrics.strong_face_down)
        self.assertTrue(metrics.is_face_down)
        self.assertGreater(metrics.risk_score, 0.0)

    def test_pose_metrics_detects_side_pose(self):
        from types import SimpleNamespace

        notifier = MagicMock()
        registry = MagicMock()
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            face_down_margin=20.0,
            risk_threshold=0.7,
            side_offset_ratio=0.3,
        )

        points = [
            (40, 40),  # nose shifted to the side
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (-10, 20),
            (10, 20),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (-12, 40),
            (12, 40),
        ]
        confs = [0.9, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9] + [0.8] * 6
        keypoints = SimpleNamespace(xy=points, conf=confs)

        metrics = monitor._compute_pose_metrics(points, confs, keypoints)
        self.assertTrue(metrics.is_side)
        self.assertFalse(metrics.is_face_down)
        self.assertLess(metrics.risk_score, monitor._risk_threshold)

    def test_orientation_inverted_sets_face_down(self):
        from types import SimpleNamespace

        notifier = MagicMock()
        registry = MagicMock()
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            face_down_margin=15.0,
            risk_threshold=0.7,
        )

        points = [
            (35, 50),  # nose bem abaixo dos ombros
            (35, 48),
            (35, 48),
            (35, 47),
            (35, 47),
            (10, 20),   # left_shoulder
            (60, 20),   # right_shoulder
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (15, 40),   # left_hip
            (65, 40),   # right_hip
        ]
        confs = [0.9] * len(points)
        keypoints = SimpleNamespace(xy=points, conf=confs)

        metrics = monitor._compute_pose_metrics(points, confs, keypoints)
        self.assertTrue(metrics.orientation_inverted)
        self.assertTrue(metrics.strong_face_down)
        self.assertTrue(metrics.is_face_down)

    def test_orientation_inverted_requires_extra_evidence(self):
        from types import SimpleNamespace

        notifier = MagicMock()
        registry = MagicMock()
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            face_down_margin=15.0,
            risk_threshold=0.8,
        )

        points = [
            (35, 22),  # nose only slightly below shoulders
            (35, 22),
            (35, 22),
            (35, 21),
            (35, 21),
            (10, 20),   # left_shoulder
            (60, 20),   # right_shoulder
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (15, 40),
            (65, 40),
        ]
        confs = [0.9] * len(points)
        keypoints = SimpleNamespace(xy=points, conf=confs)

        metrics = monitor._compute_pose_metrics(points, confs, keypoints)
        self.assertTrue(metrics.orientation_inverted)
        self.assertFalse(metrics.strong_face_down)
        self.assertFalse(metrics.is_face_down)

    def test_orientation_inverted_with_low_face_confidence_triggers_risk(self):
        from types import SimpleNamespace

        notifier = MagicMock()
        registry = MagicMock()
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            face_down_margin=15.0,
            risk_threshold=0.7,
        )

        points = [
            (35, 22),
            (35, 22),
            (35, 22),
            (35, 21),
            (35, 21),
            (10, 20),
            (60, 20),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (15, 40),
            (65, 40),
        ]
        confs = [0.05] * 5 + [0.9] * (len(points) - 5)
        keypoints = SimpleNamespace(xy=points, conf=confs)

        metrics = monitor._compute_pose_metrics(points, confs, keypoints)
        self.assertTrue(metrics.orientation_inverted)
        self.assertFalse(metrics.strong_face_down)
        self.assertTrue(metrics.is_face_down)

    def test_side_pose_triggers_suspected_notification(self):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            no_face_frames_threshold=2,
        )

        res = MagicMock()
        kps = MagicMock()
        kps.xy = [
            (0, 20),   # nose
            (0, 19),
            (0, 19),
            (0, 18),
            (0, 18),
            (-5, 20),  # shoulders nearly aligned (side pose)
            (5, 20),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (-4, 40),
            (4, 40),
        ]
        kps.conf = [0.9] * len(kps.xy)
        res.keypoints = kps

        monitor.handle_pose([res])

        notifier.notify.assert_called_once_with(
            "tok",
            title="Posição suspeita",
            message="Bebê de lado — monitorando possível virada",
            level="Importante",
        )
        self.assertTrue(monitor.face_down_suspected_sent)

    def test_side_motion_does_not_trigger_alert(self):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            no_face_frames_threshold=10,
        )

        def make_result(nose_x: float):
            res = MagicMock()
            kps = MagicMock()
            kps.xy = [
                (nose_x, 20),
                (nose_x, 19),
                (nose_x, 19),
                (nose_x, 18),
                (nose_x, 18),
                (-5, 20),
                (5, 20),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (-4, 40),
                (4, 40),
            ]
            kps.conf = [0.9] * len(kps.xy)
            res.keypoints = kps
            return res

        for nose in (6, -6, 6, -6, 6, -6):
            monitor.handle_pose([make_result(nose)])

        notifier.notify.assert_not_called()

    def test_missing_pose_keeps_face_down_flag(self):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        monitor = PositionMonitor(
            notifier,
            registry,
            model=MagicMock(),
            face_down_margin=0,
        )

        face_down = MagicMock()
        kps = MagicMock()
        kps.xy = [
            (0, 50),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (-40, 20),
            (40, 20),
        ]
        face_down.keypoints = kps

        monitor.handle_pose([face_down])
        self.assertTrue(monitor.face_down_sent)

        monitor.handle_pose([])
        self.assertTrue(monitor.face_down_sent)


if __name__ == "__main__":
    unittest.main()
