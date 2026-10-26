import builtins
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from axell_export.config import get_preset
from axell_export.exporting import EXPECTED_RUNTIME, _runtime_loader, export_engine, plan_export


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.weights = self.root / "weights.pt"
        self.weights.write_bytes(b"opaque checkpoint fixture")
        self.output = self.root / "run"
        self.config = get_preset("fp16-640-nms")
        self.calls = []

    def loader(self, checkpoint):
        outer = self

        class FakeModel:
            def export(self, **kwargs):
                outer.calls.append((Path(checkpoint), kwargs))
                path = Path(checkpoint).with_suffix(".engine")
                path.write_bytes(b"opaque exported engine fixture")
                return str(path)

        return FakeModel()

    def test_plan_does_not_create_output_or_import_a_model_runtime(self):
        original_import = builtins.__import__

        def import_without_runtime(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "ultralytics", "tensorrt"}:
                raise AssertionError(f"unexpected runtime import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_runtime):
            plan = plan_export(self.weights, self.output, self.config)
        self.assertEqual(plan["export_kwargs"]["imgsz"], (640, 640))
        self.assertIsNone(plan["calibration"])
        self.assertFalse(self.output.exists())

    def test_success_uses_private_workcopy_without_copying_engine_onto_itself(self):
        original = self.weights.read_bytes()
        manifest = export_engine(self.weights, self.output, self.config, model_loader=self.loader)
        document = json.loads(manifest.read_text())
        self.assertEqual(manifest, self.output / "export.json")
        self.assertEqual(document["status"], "complete")
        self.assertEqual(document["backend"], "injected")
        self.assertEqual(document["source_weights_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(
            document["engine"]["sha256"],
            hashlib.sha256((self.output / "model.engine").read_bytes()).hexdigest(),
        )
        self.assertEqual(self.calls[0][0], self.output / "model.pt")
        self.assertFalse((self.output / "model.pt").exists())
        self.assertFalse(self.weights.with_suffix(".engine").exists())
        self.assertEqual(self.weights.read_bytes(), original)

    def test_nms_overrides_reach_backend_and_manifest(self):
        config = replace(self.config, conf=0.001, iou=0.6, max_det=5)
        manifest = export_engine(self.weights, self.output, config, model_loader=self.loader)
        document = json.loads(manifest.read_text())
        for field, expected in (("conf", 0.001), ("iou", 0.6), ("max_det", 5)):
            self.assertEqual(self.calls[0][1][field], expected)
            self.assertEqual(document["export_kwargs"][field], expected)
            self.assertEqual(document["requested_config"][field], expected)

    def test_existing_output_is_never_reused(self):
        self.output.mkdir()
        sentinel = self.output / "model.engine"
        sentinel.write_bytes(b"keep prior result")
        with self.assertRaises(FileExistsError):
            export_engine(self.weights, self.output, self.config, model_loader=self.loader)
        self.assertEqual(sentinel.read_bytes(), b"keep prior result")
        self.assertEqual(self.calls, [])

    def test_missing_or_empty_checkpoint_is_rejected_before_backend(self):
        for name, contents in (("missing.pt", None), ("empty.pt", b""), ("invalid.bin", b"x")):
            path = self.root / name
            if contents is not None:
                path.write_bytes(contents)
            with self.subTest(name=name), self.assertRaises(ValueError):
                export_engine(path, self.output, self.config, model_loader=self.loader)
            self.assertFalse(self.output.exists())

    def test_invalid_device_is_rejected(self):
        for device in (-1, 1, True, "cpu"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                plan_export(self.weights, self.output, self.config, device=device)

    def test_backend_failure_is_recorded_and_workcopy_is_removed(self):
        class FailingModel:
            def export(self, **kwargs):
                raise RuntimeError("simulated engine build failure")

        with self.assertRaisesRegex(RuntimeError, "simulated"):
            export_engine(
                self.weights, self.output, self.config, model_loader=lambda _: FailingModel()
            )
        document = json.loads((self.output / "export.json").read_text())
        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["error"]["type"], "RuntimeError")
        self.assertNotIn("engine", document)
        self.assertFalse((self.output / "model.pt").exists())

    def test_wrong_engine_return_path_is_rejected_without_searching_for_candidates(self):
        foreign = self.root / "unrelated.engine"
        foreign.write_bytes(b"unrelated")
        outer = self

        class WrongModel:
            def export(self, **kwargs):
                (outer.output / "model.engine").write_bytes(b"otherwise valid")
                return foreign

        with self.assertRaisesRegex(RuntimeError, "unexpected engine path"):
            export_engine(
                self.weights, self.output, self.config, model_loader=lambda _: WrongModel()
            )
        self.assertEqual(json.loads((self.output / "export.json").read_text())["status"], "failed")
        self.assertEqual(foreign.read_bytes(), b"unrelated")

    def test_empty_or_missing_engine_never_gets_a_complete_manifest(self):
        for create_empty in (False, True):
            output = self.root / str(create_empty)

            class EmptyModel:
                def export(self, **kwargs):
                    path = output / "model.engine"
                    if create_empty:
                        path.touch()
                    return path

            with self.subTest(create_empty=create_empty), self.assertRaises(RuntimeError):
                export_engine(
                    self.weights, output, self.config, model_loader=lambda _: EmptyModel()
                )
            self.assertEqual(json.loads((output / "export.json").read_text())["status"], "failed")

    def test_runtime_check_failure_does_not_create_output(self):
        with patch("axell_export.exporting._runtime_loader", side_effect=RuntimeError("no CUDA")):
            with self.assertRaisesRegex(RuntimeError, "no CUDA"):
                export_engine(self.weights, self.output, self.config)
        self.assertFalse(self.output.exists())

    def test_int8_requires_train_only_calibration_and_records_exact_arguments(self):
        training = self.root / "train.jpg"
        training.write_bytes(b"opaque training image placeholder")
        calibration = self.root / "data.yaml"
        calibration.write_text(
            yaml.safe_dump(
                {
                    "train": "train.jpg",
                    "val": "nonexistent-validation",
                    "test": "nonexistent-test",
                    "names": ["object"],
                }
            )
        )
        config = replace(get_preset("int8-512x768-raw"), calibration_fraction=1.0)
        preview = plan_export(self.weights, self.output, config, calibration_data=calibration)
        self.assertFalse(self.output.exists())
        manifest = export_engine(
            self.weights,
            self.output,
            config,
            calibration_data=calibration,
            model_loader=self.loader,
        )
        document = json.loads(manifest.read_text())
        kwargs = self.calls[0][1]
        self.assertEqual(kwargs, preview["export_kwargs"])
        self.assertEqual(kwargs["split"], "train")
        self.assertEqual(kwargs["fraction"], 1.0)
        self.assertTrue(kwargs["int8"])
        self.assertFalse(kwargs["half"])
        staged = yaml.safe_load(Path(kwargs["data"]).read_text())
        self.assertNotIn("test", staged)
        self.assertEqual(Path(staged["train"]).read_text().splitlines(), [str(training)])
        self.assertEqual(document["calibration"]["image_count"], 1)
        self.assertEqual(document["calibration"]["effective_image_count"], 1)

    def test_selected_paths_hashes_and_order_match_plan_manifest_and_staging(self):
        images = self.root / "images"
        images.mkdir()
        for index in range(10):
            (images / f"{index}.png").write_bytes(bytes([index]))
        source = self.root / "data.yaml"
        source.write_text("train: images\nnames: [object]\n")
        config = get_preset("int8-512x768-raw")
        preview = plan_export(self.weights, self.output, config, calibration_data=source)
        manifest = export_engine(
            self.weights, self.output, config, calibration_data=source, model_loader=self.loader
        )
        document = json.loads(manifest.read_text())
        self.assertEqual(document["calibration"], preview["calibration"])
        records = document["calibration"]["selected_images"]
        self.assertEqual(len(records), 3)
        self.assertEqual(
            (self.output / "calibration-train.txt").read_text().splitlines(),
            [record["path"] for record in records],
        )
        for record in records:
            self.assertEqual(
                record["sha256"], hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
            )

    def test_content_change_during_preflight_is_rejected_without_creating_output(self):
        image = self.root / "train.png"
        image.write_bytes(b"original")
        source = self.root / "data.yaml"
        source.write_text("train: train.png\nnames: [object]\n")
        config = replace(get_preset("int8-512x768-raw"), calibration_fraction=1.0)

        def runtime_loader(device):
            image.write_bytes(b"changed")
            return self.loader, {}

        with patch("axell_export.exporting._runtime_loader", side_effect=runtime_loader):
            with self.assertRaisesRegex(ValueError, "inputs changed during preflight"):
                export_engine(self.weights, self.output, config, calibration_data=source)
        self.assertFalse(self.output.exists())

    def test_content_change_during_export_marks_run_failed(self):
        image = self.root / "train.png"
        image.write_bytes(b"original")
        source = self.root / "data.yaml"
        source.write_text("train: train.png\nnames: [object]\n")
        config = replace(get_preset("int8-512x768-raw"), calibration_fraction=1.0)

        def changed_loader(path):
            image.write_bytes(b"changed during export")
            return self.loader(path)

        with self.assertRaisesRegex(ValueError, "content changed"):
            export_engine(
                self.weights,
                self.output,
                config,
                calibration_data=source,
                model_loader=changed_loader,
            )
        self.assertEqual(json.loads((self.output / "export.json").read_text())["status"], "failed")

    def test_fraction_that_rounds_to_zero_is_rejected_without_creating_output(self):
        training = self.root / "train.jpg"
        training.write_bytes(b"opaque image placeholder")
        calibration = self.root / "data.yaml"
        calibration.write_text(yaml.safe_dump({"train": "train.jpg", "names": ["object"]}))
        with self.assertRaisesRegex(ValueError, "zero images"):
            export_engine(
                self.weights,
                self.output,
                get_preset("int8-512x768-raw"),
                calibration_data=calibration,
                model_loader=self.loader,
            )
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls, [])

    def test_production_backend_installs_readonly_calibration_adapter_before_export(self):
        training = self.root / "train.jpg"
        training.write_bytes(b"opaque image placeholder")
        calibration = self.root / "data.yaml"
        calibration.write_text(yaml.safe_dump({"train": "train.jpg", "names": ["object"]}))
        config = replace(get_preset("int8-512x768-raw"), calibration_fraction=1.0)
        model = self.loader(self.output / "model.pt")
        original_export = model.export
        with (
            patch("axell_export.exporting._runtime_loader", return_value=(lambda _: model, {})),
            patch("axell_export.runtime.attach_calibration_loader") as attach,
        ):

            def export_after_attachment(**kwargs):
                attach.assert_called_once()
                self.assertIs(attach.call_args.args[0], model)
                self.assertEqual(attach.call_args.args[1].images, (training,))
                self.assertEqual(attach.call_args.args[2], config)
                return original_export(**kwargs)

            model.export = export_after_attachment
            manifest = export_engine(
                self.weights, self.output, config, calibration_data=calibration
            )
        document = json.loads(manifest.read_text())
        self.assertEqual(document["calibration"]["loader"], "readonly_rgb_letterbox")
        self.assertEqual(document["calibration"]["tensor_shape"], [1, 3, 512, 768])
        self.assertEqual(document["backend"], "ultralytics")

    def test_calibration_usage_must_match_precision(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            plan_export(self.weights, self.output, get_preset("int8-512x768-raw"))
        with self.assertRaisesRegex(ValueError, "only accepted"):
            plan_export(
                self.weights, self.output, self.config, calibration_data=self.root / "data.yaml"
            )
        self.assertFalse(self.output.exists())

    def test_symlink_output_directory_is_rejected(self):
        self.output.symlink_to(self.root / "missing-directory", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            plan_export(self.weights, self.output, self.config)


class RuntimePreflightTests(unittest.TestCase):
    def setUp(self):
        self.versions = {
            **EXPECTED_RUNTIME,
            "torch": "2.4.0+cu121",
            "torchvision": "0.19.0+cu121",
            "onnxruntime-gpu": "1.22.0",
        }
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda _: "fixture GPU",
            ),
            version=SimpleNamespace(cuda="12.1"),
        )
        self.loader = object()
        self.modules = {
            "torch": self.torch,
            "ultralytics": SimpleNamespace(YOLO=self.loader),
        }

    def test_preflight_checks_versions_and_disables_automatic_installation(self):
        with (
            patch("axell_export.exporting.metadata.version", side_effect=self.versions.__getitem__),
            patch.dict(sys.modules, self.modules),
            patch.dict(os.environ, {}, clear=False),
        ):
            loader, environment = _runtime_loader(0)
            self.assertIs(loader, self.loader)
            self.assertEqual(environment["gpu"], "fixture GPU")
            self.assertEqual(environment["onnxruntime-gpu"], "1.22.0")
            self.assertEqual(os.environ["YOLO_AUTOINSTALL"], "false")
            self.assertEqual(os.environ["YOLO_OFFLINE"], "true")

    def test_dependency_mismatch_stops_before_loading_a_model(self):
        self.versions["ultralytics"] = "0.0.0"
        with patch(
            "axell_export.exporting.metadata.version", side_effect=self.versions.__getitem__
        ):
            with self.assertRaisesRegex(RuntimeError, "8.3.170"):
                _runtime_loader(0)

    def test_unavailable_cuda_is_rejected(self):
        self.torch.cuda.is_available = lambda: False
        with (
            patch("axell_export.exporting.metadata.version", side_effect=self.versions.__getitem__),
            patch.dict(sys.modules, self.modules),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA device 0"):
                _runtime_loader(0)

    def test_preflight_leaves_tensor_rt_import_to_the_exporter(self):
        original_import = builtins.__import__

        def import_before_onnx(name, *args, **kwargs):
            if name.split(".")[0] == "tensorrt":
                raise AssertionError("TensorRT must not be imported before ONNX conversion")
            return original_import(name, *args, **kwargs)

        with (
            patch("axell_export.exporting.metadata.version", side_effect=self.versions.__getitem__),
            patch.dict(sys.modules, self.modules),
            patch("builtins.__import__", side_effect=import_before_onnx),
        ):
            _, environment = _runtime_loader(0)
        self.assertEqual(environment["tensorrt-cu12-bindings"], "10.13.0.35")


if __name__ == "__main__":
    unittest.main()
