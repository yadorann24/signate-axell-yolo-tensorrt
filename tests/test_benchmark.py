import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from axell_export.benchmark import (
    BenchmarkConfig, _export_context, _image_inventory, _read_image, benchmark_model, main,
)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.weights = self.root / "local.pt"
        self.weights.write_bytes(b"synthetic checkpoint placeholder")
        self.image = self.root / "image.png"
        cv2.imwrite(str(self.image), np.full((40, 80, 3), 127, dtype=np.uint8))
        self.images = self.root / "images.txt"
        self.images.write_text("image.png\n")
        self.config = BenchmarkConfig((64, 96), "cpu", 2, 3)

    def test_timing_excludes_warmup_and_records_top_five_original_coordinates(self):
        events = []

        class Model:
            names = {0: "synthetic"}

            def predict(inner, source, **kwargs):
                events.append("predict")
                self.assertEqual(kwargs["imgsz"], [64, 96])
                self.assertFalse(kwargs["rect"])
                self.assertFalse(kwargs["save"])
                boxes = torch.tensor([[1, 2, 30, 35, p, 0] for p in (.2, .8, .3, .9, .5, .1)])
                return [Results(source, "synthetic.png", inner.names, boxes=boxes)]

        def loader(*args):
            return Model(), lambda: events.append("sync"), {"device": "fake"}

        with patch("axell_export.benchmark.time.perf_counter_ns", side_effect=[0, 1000000] * 3):
            path = benchmark_model(self.weights, self.images, self.root / "run", self.config,
                                   runtime_loader=loader)
        report = json.loads(path.read_text())
        self.assertEqual(report["backend"], "injected")
        self.assertEqual(report["latency_ms"]["count"], 3)
        self.assertEqual(report["latency_ms"]["mean"], 1)
        self.assertEqual(events, ["predict", "predict", "sync"] + ["sync", "predict", "sync"] * 3)
        predictions = report["predictions"][0]
        self.assertEqual((predictions["height"], predictions["width"]), (40, 80))
        self.assertEqual(len(predictions["detections"]), 5)
        self.assertAlmostEqual(predictions["detections"][0]["confidence"], .9)
        self.assertEqual(predictions["detections"][0]["xyxy"], [1, 2, 30, 35])
        with self.assertRaises(FileExistsError):
            benchmark_model(self.weights, self.images, self.root / "run", self.config)

    def test_image_identity_duplicates_urls_and_changed_bytes(self):
        inventory = _image_inventory(self.images)
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            _read_image(inventory[0])
        for content in ("image.png\n./image.png\n", "https://example.org/image.png", "", "missing.png"):
            with self.subTest(content=content):
                self.images.write_text(content)
                with self.assertRaises(ValueError):
                    _image_inventory(self.images)

    def test_failure_is_recorded(self):
        def loader(*args):
            raise RuntimeError("synthetic failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            benchmark_model(self.weights, self.images, self.root / "failed", self.config,
                            runtime_loader=loader)
        report = json.loads((self.root / "failed/benchmark.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "RuntimeError")

    def test_embedded_nms_must_match_and_engine_hash_is_checked(self):
        engine = self.root / "model.engine"
        engine.write_bytes(b"synthetic engine placeholder")
        manifest = self.root / "export.json"
        document = {
            "status": "complete", "backend": "ultralytics",
            "engine": {"sha256": hashlib.sha256(engine.read_bytes()).hexdigest()},
            "requested_config": {"image_size": [64, 96], "batch_size": 1, "nms": True,
                                 "conf": .25, "iou": .7, "max_det": 300},
        }
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "embedded NMS"):
            _export_context(engine, BenchmarkConfig((64, 96)), manifest)
        correct = BenchmarkConfig((64, 96), conf=.25)
        self.assertIsNotNone(_export_context(engine, correct, manifest))
        for field in ("conf", "iou", "max_det"):
            del document["requested_config"][field]
        manifest.write_text(json.dumps(document))
        self.assertIsNotNone(_export_context(engine, correct, manifest))
        engine.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            _export_context(engine, correct, manifest)

    def test_invalid_configuration_and_cli(self):
        for kwargs in ({"image_size": (33, 64)}, {"warmup": 0}, {"repeats": True},
                       {"max_det": -1}, {"conf": float("nan")}, {"iou": -.1},
                       {"device": "cpu", "half": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BenchmarkConfig(**kwargs).validate()
        with patch("axell_export.benchmark.benchmark_model", return_value=self.root / "report") as run:
            main(["--weights", str(self.weights), "--images", str(self.images),
                  "--output-dir", str(self.root / "cli"), "--device", "cpu", "--image-size", "64", "96"])
        self.assertEqual(run.call_args.args[3].image_size, (64, 96))

    def test_real_cpu_prediction_with_random_checkpoint(self):
        # This validates the actual pinned YOLO predictor, not model accuracy.
        torch.set_num_threads(1)
        model = YOLO("yolo11n.yaml")
        model.save(self.weights)
        with patch("ultralytics.utils.downloads.safe_download",
                   side_effect=AssertionError("test must not download weights")):
            path = benchmark_model(self.weights, self.images, self.root / "real",
                                   BenchmarkConfig((64, 96), "cpu", 1, 1))
        report = json.loads(path.read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["backend"], "ultralytics")
        self.assertGreater(report["latency_ms"]["mean"], 0)
        self.assertEqual(report["predict_kwargs"]["imgsz"], [64, 96])
        self.assertEqual(report["images"][0]["sha256"], hashlib.sha256(self.image.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
