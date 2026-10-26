"""Construct the registered architectures using random weights and synthetic input."""

import gc
from importlib.util import find_spec
import os
import tempfile
import unittest
from unittest.mock import patch

from axell_export.registry import build_model, list_models


@unittest.skipUnless(
    find_spec("torch") is not None and find_spec("ultralytics") is not None,
    "Model integration tests require the runtime extra",
)
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_directory = tempfile.TemporaryDirectory(prefix="axell-model-tests-")
        cls.addClassCleanup(cls.config_directory.cleanup)
        cls.environment = patch.dict(
            os.environ,
            {
                "YOLO_CONFIG_DIR": cls.config_directory.name,
                "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true",
            },
        )
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)

        import torch

        cls.torch = torch
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, original_threads)

    def assert_detector_output(self, name, num_classes):
        """Exercise the public factory without allowing checkpoint downloads."""
        with (
            patch(
                "ultralytics.utils.downloads.safe_download",
                side_effect=AssertionError("Model construction must not download files"),
            ),
            patch(
                "torch.hub.download_url_to_file",
                side_effect=AssertionError("Model construction must not download weights"),
            ),
        ):
            model = build_model(name, num_classes=num_classes)
            try:
                self.assertEqual(model.yaml["nc"], num_classes)
                self.assertEqual(model.model[-1].nc, num_classes)
                self.assertEqual(model.stride.tolist(), [8.0, 16.0, 32.0])
                model.eval()
                with self.torch.inference_mode():
                    predictions, feature_maps = model(self.torch.zeros(1, 3, 256, 256))
                # Four box coordinates and one probability per class, before NMS.
                self.assertEqual(tuple(predictions.shape), (1, num_classes + 4, 1344))
                self.assertTrue(self.torch.isfinite(predictions).all().item())
                self.assertEqual(len(feature_maps), 3)
            finally:
                del model
                gc.collect()

    def test_all_eight_architectures_support_91_class_inference(self):
        models = list_models()
        self.assertEqual(len(models), 8)
        for model in models:
            with self.subTest(model=model.name):
                self.assert_detector_output(model.name, 91)

    def test_class_count_changes_detector_output(self):
        self.assert_detector_output("yolo11n", 7)


if __name__ == "__main__":
    unittest.main()
