import unittest
from unittest.mock import MagicMock, patch, call

from src.monitor.position_monitor import PositionMonitor


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
            (0, 20),  # left_shoulder
            (0, 20),  # right_shoulder
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
            (0, 20),
            (0, 20),
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

        notifier = MagicMock()
        registry = MagicMock()
        model = MagicMock()
        monitor = PositionMonitor(notifier, registry, model=model)

        plotted_frame = object()
        result = MagicMock()
        result.plot.return_value = plotted_frame
        model.return_value = [result]

        monitor.analyze_frame(object(), show=True)
        monitor.analyze_frame(object(), show=True)

        expected_flag = cv2_mock.WINDOW_NORMAL | cv2_mock.WINDOW_GUI_EXPANDED
        cv2_mock.namedWindow.assert_called_once_with("Pose", expected_flag)
        self.assertEqual(
            cv2_mock.imshow.call_args_list,
            [call("Pose", plotted_frame)] * 2,
        )
        self.assertEqual(
            cv2_mock.waitKey.call_args_list,
            [call(1)] * 2,
        )

    @patch("src.monitor.position_monitor.cv2")
    def test_pose_window_updates_without_results(self, cv2_mock):
        cv2_mock.WINDOW_NORMAL = 1
        cv2_mock.WINDOW_GUI_EXPANDED = 2

        notifier = MagicMock()
        registry = MagicMock()
        model = MagicMock(return_value=[])
        monitor = PositionMonitor(notifier, registry, model=model)

        frame = object()
        monitor.analyze_frame(frame, show=True)

        cv2_mock.namedWindow.assert_called_once()
        cv2_mock.imshow.assert_called_once_with("Pose", frame)
        cv2_mock.waitKey.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
