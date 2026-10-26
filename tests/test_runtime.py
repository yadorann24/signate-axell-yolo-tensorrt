"""Exercise the real pinned exporter callback with generated calibration images."""

from collections import defaultdict
from dataclasses import replace
import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axell_export.calibration import inspect_calibration_data, select_calibration_images
from axell_export.config import get_preset
from axell_export.runtime import attach_calibration_loader


class CallbackModel:
    def __init__(self):
        self.callbacks = defaultdict(list)

    def add_callback(self, event, callback):
        self.callbacks[event].append(callback)


@unittest.skipUnless(
    find_spec("torch") is not None and find_spec("ultralytics") is not None,
    "Runtime integration tests require the runtime extra",
)
class CalibrationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_directory = tempfile.TemporaryDirectory(prefix="axell-runtime-settings-")
        cls.addClassCleanup(cls.config_directory.cleanup)
        environment = patch.dict(
            os.environ,
            {
                "YOLO_CONFIG_DIR": cls.config_directory.name,
                "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true",
            },
        )
        environment.start()
        cls.addClassCleanup(environment.stop)
        import torch
        from ultralytics.engine.exporter import Exporter

        cls.torch = torch
        cls.Exporter = Exporter
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, original_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="axell-runtime-images-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "train"
        self.images.mkdir()
        self.source = self.root / "dataset.yaml"
        self.source.write_text(
            "train: train\nval: missing-val\ntest: missing-test\nnames: [object]\n"
        )
        self.config = replace(get_preset("int8-512x768-raw"), calibration_fraction=1.0)

    def make_image(self, name="image.png", color=(250, 12, 4), size=(96, 64)):
        from PIL import Image

        path = self.images / name
        Image.new("RGB", size, color).save(path)
        return path

    def exporter_with_adapter(self, config=None, calibration=None):
        config = config or self.config
        model = CallbackModel()
        attach_calibration_loader(
            model, calibration or inspect_calibration_data(self.source), config
        )
        exporter = self.Exporter(
            overrides={**config.export_kwargs(), "data": str(self.source), "split": "train"},
            _callbacks=model.callbacks,
        )
        exporter.imgsz = list(config.image_size)
        exporter.device = self.torch.device("cpu")
        return exporter

    def test_rectangular_batches_are_rgb_uint8_without_vendor_dataset(self):
        self.make_image()
        exporter = self.exporter_with_adapter()
        with (
            patch.object(
                self.Exporter,
                "get_int8_calibration_dataloader",
                side_effect=AssertionError("The square vendor loader must not run"),
            ),
            patch(
                "ultralytics.engine.exporter.check_det_dataset",
                side_effect=AssertionError("Dataset checks must not access val or test"),
            ),
            patch(
                "ultralytics.data.utils.check_font",
                side_effect=AssertionError("Calibration must not request fonts"),
            ),
        ):
            exporter.run_callbacks("on_export_start")
            loader = exporter.get_int8_calibration_dataloader("TensorRT:")
            batch = next(iter(loader))["img"]
        self.assertEqual(loader.batch_size, 1)
        self.assertEqual(tuple(batch.shape), (1, 3, 512, 768))
        self.assertEqual(batch.dtype, self.torch.uint8)
        self.assertEqual(batch[0, :, 256, 384].tolist(), [250, 12, 4])
        # Standard YOLO inference enlarges a small source to the target dimensions.
        self.assertEqual(batch[0, :, 0, 0].tolist(), [250, 12, 4])
        self.assertEqual(batch[0, :, -1, -1].tolist(), [250, 12, 4])

    def test_aspect_ratio_is_preserved_with_gray_padding(self):
        self.make_image(size=(96, 96))
        exporter = self.exporter_with_adapter()
        exporter.run_callbacks("on_export_start")
        batch = next(iter(exporter.get_int8_calibration_dataloader()))["img"]
        self.assertEqual(tuple(batch.shape), (1, 3, 512, 768))
        self.assertEqual(batch[0, :, 256, 127].tolist(), [114, 114, 114])
        self.assertEqual(batch[0, :, 256, 128].tolist(), [250, 12, 4])
        self.assertEqual(batch[0, :, 256, 639].tolist(), [250, 12, 4])
        self.assertEqual(batch[0, :, 256, 640].tolist(), [114, 114, 114])

    def test_fraction_selects_sorted_training_images_once(self):
        for index in (4, 2, 0, 3, 1):
            self.make_image(f"{index}.png", color=(index + 1, 0, 0))
        config = replace(
            self.config, calibration_fraction=0.5, calibration_selection="sorted-first"
        )
        exporter = self.exporter_with_adapter(config)
        exporter.run_callbacks("on_export_start")
        batches = list(exporter.get_int8_calibration_dataloader())
        self.assertEqual(len(batches), round(5 * 0.5))
        self.assertEqual([batch["img"][0, 0, 256, 384].item() for batch in batches], [1, 2])

    def test_seeded_plan_selection_matches_runtime_and_is_not_fractioned_twice(self):
        for index in range(10):
            self.make_image(f"{index}.png", color=(index + 1, 0, 0))
        config = replace(self.config, calibration_fraction=0.3, calibration_seed=7)
        selection = select_calibration_images(inspect_calibration_data(self.source), config)
        exporter = self.exporter_with_adapter(config, selection)
        exporter.run_callbacks("on_export_start")
        batches = list(exporter.get_int8_calibration_dataloader())
        self.assertEqual(len(batches), 3)
        self.assertEqual(
            [batch["img"][0, 0, 256, 384].item() for batch in batches],
            [int(path.stem) + 1 for path in selection.images],
        )

    def test_content_change_after_adapter_creation_is_rejected_before_decode(self):
        image = self.make_image()
        exporter = self.exporter_with_adapter()
        exporter.run_callbacks("on_export_start")
        image.write_bytes(b"modified bytes")
        with self.assertRaisesRegex(ValueError, "content changed"):
            next(iter(exporter.get_int8_calibration_dataloader()))

    def test_content_change_during_decode_is_rejected(self):
        import cv2

        image = self.make_image()
        exporter = self.exporter_with_adapter()
        exporter.run_callbacks("on_export_start")
        decode = cv2.imread

        def changed_after_decode(path, flags):
            result = decode(path, flags)
            image.write_bytes(image.read_bytes() + b"changed")
            return result

        with patch("cv2.imread", side_effect=changed_after_decode):
            with self.assertRaisesRegex(ValueError, "content changed"):
                next(iter(exporter.get_int8_calibration_dataloader()))

    def test_selection_for_a_different_seed_is_rejected(self):
        self.make_image()
        selection = select_calibration_images(inspect_calibration_data(self.source), self.config)
        with self.assertRaisesRegex(ValueError, "selection does not match"):
            self.exporter_with_adapter(replace(self.config, calibration_seed=1), selection)

    def test_real_nms_export_wrapper_uses_thresholds_and_detection_limit(self):
        from ultralytics.engine.exporter import NMSModel

        torch = self.torch

        class SyntheticDetections(torch.nn.Module):
            task = "detect"
            names = {0: "object"}

            def forward(self, inputs):
                return torch.tensor(
                    [
                        [0, 0, 20, 20, 0.9],
                        [1, 1, 21, 21, 0.8],
                        [40, 40, 50, 50, 0.1],
                    ]
                ).T.unsqueeze(0)

        for conf, iou, max_det, count in (
            (0.25, 0.5, 3, 1),
            (0.01, 0.5, 3, 2),
            (0.01, 0.99, 3, 3),
            (0.01, 0.99, 1, 1),
        ):
            with self.subTest(conf=conf, iou=iou, max_det=max_det):
                config = replace(get_preset("fp16-640-nms"), conf=conf, iou=iou, max_det=max_det)
                args = self.Exporter(overrides=config.export_kwargs()).args
                output = NMSModel(SyntheticDetections(), args)(torch.zeros(1, 3, 64, 64))
                self.assertEqual(output.shape, (1, max_det, 6))
                self.assertEqual(int((output[0, :, 4] > 0).sum()), count)

    def test_readable_truncated_jpeg_is_not_repaired_and_no_cache_is_written(self):
        image = self.make_image("truncated.jpg")
        image.write_bytes(image.read_bytes()[:-2])

        def snapshot():
            return {
                str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*")
                if path.is_file()
            }

        before = snapshot()
        exporter = self.exporter_with_adapter()
        exporter.run_callbacks("on_export_start")
        batch = next(iter(exporter.get_int8_calibration_dataloader()))["img"]
        self.assertEqual(tuple(batch.shape), (1, 3, 512, 768))
        self.assertEqual(snapshot(), before)

    def test_unreadable_training_image_fails_explicitly(self):
        (self.images / "broken.png").write_bytes(b"not an image")
        exporter = self.exporter_with_adapter()
        exporter.run_callbacks("on_export_start")
        with self.assertRaisesRegex(ValueError, "Could not decode calibration"):
            next(iter(exporter.get_int8_calibration_dataloader()))

    def test_exporter_shape_mismatch_is_rejected_before_loading_images(self):
        self.make_image()
        exporter = self.exporter_with_adapter()
        exporter.imgsz = [512, 512]
        with self.assertRaisesRegex(ValueError, "input shape"):
            exporter.run_callbacks("on_export_start")

    def test_zero_selected_images_are_rejected(self):
        self.make_image()
        with self.assertRaisesRegex(ValueError, "zero images"):
            self.exporter_with_adapter(replace(self.config, calibration_fraction=0.3))


if __name__ == "__main__":
    unittest.main()
