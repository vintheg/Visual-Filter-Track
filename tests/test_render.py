import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import main


class RenderTests(unittest.TestCase):
    def args(self, source, output, **kwargs):
        values = dict(input=source, output=output, preset='cyber', seed=7,
                      scale=1.0, limit_frames=0, preview=False)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_lost_optical_flow_recovers(self):
        gray = np.zeros((64, 64), np.uint8)
        state = main.empty_track_state()
        main.add_tracks(state, np.array([[20, 20], [40, 40]], np.float32))
        with patch.object(cv2, 'calcOpticalFlowPyrLK', return_value=(None, None, None)):
            main.update_track_state(state, gray, gray, main.PRESETS['cyber'], 1)
        self.assertEqual(len(state.ids), 0)
        self.assertEqual(state.raw_points.shape, (0, 2))

    def test_invalid_arguments_do_not_open_input(self):
        with patch.object(main, 'open_capture') as capture:
            for overrides in ({'scale': float('nan')}, {'scale': float('inf')},
                              {'scale': 0}, {'limit_frames': -1}, {'seed': -1}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    main.process_video(self.args(Path('in.mp4'), Path('out.mp4'), **overrides))
            with self.assertRaises(ValueError):
                main.process_video(self.args(Path('in.mp4'), Path('in.mp4')))
            capture.assert_not_called()

    def test_capture_released_if_writer_fails(self):
        with patch.object(main, 'open_capture') as opened:
            cap = opened.return_value
            cap.get.return_value = 24
            cap.read.return_value = (True, np.zeros((64, 64, 3), np.uint8))
            with patch.object(main, 'make_writer', side_effect=RuntimeError('writer failed')):
                with self.assertRaises(RuntimeError):
                    main.process_video(self.args(Path('in.mp4'), Path('out.mp4')))
            cap.release.assert_called_once()

    def test_all_presets_encode_decodable_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.mp4'
            writer = main.make_writer(source, 24, (96, 64))
            for index in range(12):
                frame = np.zeros((64, 96, 3), np.uint8)
                if index < 8:
                    cv2.rectangle(frame, (10 + index, 15), (40 + index, 45), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            for preset in main.PRESETS:
                with self.subTest(preset=preset):
                    output = Path(tmp) / f'{preset}.mp4'
                    main.process_video(self.args(source, output, preset=preset, limit_frames=10))
                    cap = cv2.VideoCapture(str(output))
                    count = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        self.assertEqual(frame.shape, (64, 96, 3))
                        count += 1
                    self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 24)
                    cap.release()
                    self.assertEqual(count, 10)


if __name__ == '__main__':
    unittest.main()
