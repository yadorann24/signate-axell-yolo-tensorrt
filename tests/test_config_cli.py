import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from axell_export import build_model, get_model_spec, get_preset, list_models, list_presets
from axell_export.cli import main


class ConfigurationTests(unittest.TestCase):
    def test_nine_presets_have_explicit_precision_and_shape(self):
        presets = list_presets()
        self.assertEqual(len(presets), 9)
        self.assertEqual(len({config.preset for config in presets}), 9)
        for config in presets:
            with self.subTest(preset=config.preset):
                kwargs = config.export_kwargs()
                self.assertEqual(kwargs["imgsz"], config.image_size)
                self.assertEqual(kwargs["batch"], 1)
                self.assertFalse(kwargs["dynamic"])
                self.assertNotEqual(kwargs["half"], kwargs["int8"])
        int8 = get_preset("int8-512x768-raw")
        self.assertFalse(int8.nms)
        self.assertEqual(int8.export_kwargs()["fraction"], 0.30)

    def test_invalid_export_config_is_rejected(self):
        baseline = get_preset("fp16-640-nms")
        invalid = (
            {"image_size": (0, 640)},
            {"image_size": (641, 640)},
            {"image_size": (640,)},
            {"image_size": (True, 640)},
            {"precision": "fp32"},
            {"nms": 1},
            {"simplify": 1},
            {"dynamic": True},
            {"batch_size": 2},
            {"workspace_gb": float("nan")},
            {"workspace_gb": 0},
            {"calibration_fraction": 0},
            {"calibration_fraction": 1.1},
            {"calibration_fraction": float("inf")},
            {"calibration_selection": "unknown"},
            {"calibration_selection": []},
            {"calibration_seed": -1},
            {"calibration_seed": True},
            {"calibration_seed": 1.2},
            {"conf": 0},
            {"conf": -0.1},
            {"conf": float("nan")},
            {"conf": True},
            {"iou": float("inf")},
            {"iou": 1.1},
            {"max_det": 0},
            {"max_det": True},
            {"max_det": 2.5},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                replace(baseline, **override).validate()

    def test_nms_parameters_are_explicit_only_for_embedded_nms(self):
        defaults = get_preset("fp16-640-nms").export_kwargs()
        self.assertEqual(
            {field: defaults[field] for field in ("conf", "iou", "max_det")},
            {"conf": 0.25, "iou": 0.7, "max_det": 300},
        )
        raw = get_preset("fp16-640-raw")
        self.assertTrue({"conf", "iou", "max_det"}.isdisjoint(raw.export_kwargs()))
        with self.assertRaisesRegex(ValueError, "NMS preset"):
            replace(raw, conf=0.01).validate()

    def test_model_names_and_invalid_construction(self):
        self.assertEqual(len(list_models()), 8)
        for model in list_models():
            self.assertEqual(get_model_spec(model.name), model)
            self.assertTrue(model.yaml_name.endswith(".yaml"))
        with self.assertRaises(ValueError):
            get_model_spec("unrecorded_model")
        with self.assertRaises(ValueError):
            build_model("yolo11n", num_classes=True)

    def test_configuration_imports_do_not_load_ml_runtime(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import axell_export,sys; "
                    "assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
                ),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class CommandTests(unittest.TestCase):
    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def test_model_and_preset_discovery(self):
        for command, count in (("list-models", 8), ("list-presets", 9)):
            code, output, error = self.invoke([command])
            self.assertEqual(code, 0, error)
            self.assertEqual(len(json.loads(output)), count)
        code, _, _ = self.invoke(["describe-preset", "--preset", "missing"])
        self.assertEqual(code, 2)

    def test_plan_creates_no_artifacts_and_rejects_ignored_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "synthetic.pt"
            weights.write_bytes(b"synthetic checkpoint used only for planning")
            output = root / "private-output"
            args = [
                "plan",
                "--weights",
                str(weights),
                "--output-dir",
                str(output),
                "--preset",
                "fp16-640-nms",
            ]
            code, text, error = self.invoke(args)
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(text)["export_kwargs"]["nms"])
            self.assertFalse(output.exists())
            code, _, _ = self.invoke(args + ["--fraction", "0.5"])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())

    def test_plan_accepts_explicit_nms_and_rejects_raw_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "synthetic.pt"
            weights.write_bytes(b"plan-only fixture")
            args = ["plan", "--weights", str(weights), "--output-dir", str(root / "run")]
            options = ["--conf", "0.001", "--iou", "0.6", "--max-det", "5"]
            code, output, error = self.invoke(args + ["--preset", "fp16-640-nms"] + options)
            self.assertEqual(code, 0, error)
            document = json.loads(output)
            for field, expected in (("conf", 0.001), ("iou", 0.6), ("max_det", 5)):
                self.assertEqual(document["requested_config"][field], expected)
                self.assertEqual(document["export_kwargs"][field], expected)
            code, _, _ = self.invoke(args + ["--preset", "fp16-640-raw"] + options)
            self.assertEqual(code, 2)

    def test_plan_records_int8_sampling_and_rejects_sampling_for_fp16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "synthetic.pt"
            weights.write_bytes(b"plan-only fixture")
            for index in range(10):
                (root / f"{index}.png").write_bytes(bytes([index]))
            source = root / "data.yaml"
            source.write_text("train: .\nnames: [object]\n")
            args = ["plan", "--weights", str(weights), "--output-dir", str(root / "run")]
            options = ["--calibration-selection", "seeded-random", "--calibration-seed", "9"]
            code, output, error = self.invoke(
                args
                + ["--preset", "int8-512x768-raw", "--calibration-data", str(source)]
                + options
                + ["--fraction", "0.5"]
            )
            self.assertEqual(code, 0, error)
            calibration = json.loads(output)["calibration"]
            self.assertEqual(calibration["selection_method"], "seeded-random")
            self.assertEqual(calibration["selection_seed"], 9)
            self.assertEqual(len(calibration["selected_images"]), 5)
            code, _, _ = self.invoke(args + ["--preset", "fp16-640-nms"] + options)
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
