import hashlib
from dataclasses import replace
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import yaml

from axell_export.calibration import (
    inspect_calibration_data,
    select_calibration_images,
    write_calibration_data,
)
from axell_export.config import get_preset


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training = self.root / "images" / "train"
        self.training.mkdir(parents=True)
        # Opaque fixtures: calibration inspection checks paths, never image contents.
        self.first = self.training / "first.jpg"
        self.second = self.training / "second.png"
        self.first.write_bytes(b"image placeholder one")
        self.second.write_bytes(b"image placeholder two")
        self.source = self.root / "data.yaml"

    def write_yaml(self, **overrides):
        document = {"path": ".", "train": "images/train", "names": ["object"]}
        document.update(overrides)
        self.source.write_text(yaml.safe_dump(document), encoding="utf-8")
        return self.source

    def test_selects_only_training_and_does_not_require_val_or_test(self):
        self.write_yaml(val="missing-validation", test="https://example.invalid/test.zip")
        data = inspect_calibration_data(self.source)
        self.assertEqual(data.images, (self.first, self.second))
        self.assertEqual(data.names, ("object",))
        self.assertEqual(data.summary()["split"], "train")
        self.assertEqual(data.summary()["image_count"], 2)
        self.assertEqual(data.source_sha256, hashlib.sha256(self.source.read_bytes()).hexdigest())
        paths = f"{self.first}\n{self.second}\n".encode("utf-8")
        self.assertEqual(data.summary()["training_paths_sha256"], hashlib.sha256(paths).hexdigest())

    def test_staged_yaml_contains_only_verified_training_list(self):
        source = self.write_yaml(val="unread-validation", test="unread-test", nc=1)
        original = source.read_bytes()
        run_dir = self.root / "run"
        run_dir.mkdir()
        staged = write_calibration_data(inspect_calibration_data(source), run_dir)
        document = yaml.safe_load(staged.read_text())
        self.assertNotIn("test", document)
        self.assertNotIn("download", document)
        self.assertEqual(document["train"], document["val"])
        self.assertEqual(document["nc"], 1)
        self.assertEqual(
            Path(document["train"]).read_text().splitlines(), [str(self.first), str(self.second)]
        )
        self.assertEqual(source.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            write_calibration_data(inspect_calibration_data(source), run_dir)

    def test_yaml_root_and_list_entries_use_their_own_directories(self):
        lists = self.root / "lists"
        lists.mkdir()
        (lists / "train.txt").write_text("../images/train/second.png\n\n", encoding="utf-8")
        configurations = self.root / "configs"
        configurations.mkdir()
        source = configurations / "data.yaml"
        source.write_text(
            yaml.safe_dump(
                {
                    "path": "..",
                    "train": "lists/train.txt",
                    "names": {0: "object"},
                    "nc": 1,
                }
            )
        )
        self.assertEqual(inspect_calibration_data(source).images, (self.second,))

    def test_duplicate_images_across_entries_are_rejected(self):
        source = self.write_yaml(train=["images/train", "images/train/first.jpg"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inspect_calibration_data(source)

    def test_symlink_alias_is_not_a_distinct_image(self):
        alias = self.training / "alias.jpg"
        alias.symlink_to(self.first)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inspect_calibration_data(self.write_yaml())

    def test_empty_training_split_or_directory_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        for train in ([], "empty"):
            with self.subTest(train=train), self.assertRaises(ValueError):
                inspect_calibration_data(self.write_yaml(train=train))

    def test_missing_training_image_is_rejected(self):
        (self.root / "train.txt").write_text("images/train/missing.jpg\n")
        with self.assertRaises(FileNotFoundError):
            inspect_calibration_data(self.write_yaml(train="train.txt"))

    def test_urls_and_download_hooks_are_rejected(self):
        cases = (
            {"download": "print('untrusted')"},
            {"download": "https://example.invalid/data.zip"},
            {"path": "https://example.invalid/data"},
            {"train": "https://example.invalid/train.txt"},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                inspect_calibration_data(self.write_yaml(**values))
        (self.root / "train.txt").write_text("https://example.invalid/a.jpg\n")
        with self.assertRaises(ValueError):
            inspect_calibration_data(self.write_yaml(train="train.txt"))

    def test_nonimage_list_entries_are_rejected(self):
        (self.root / "label.txt").write_text("annotation placeholder")
        (self.root / "train.txt").write_text("label.txt\n")
        with self.assertRaisesRegex(ValueError, "suffix"):
            inspect_calibration_data(self.write_yaml(train="train.txt"))

    def test_class_mapping_and_count_must_be_consistent(self):
        for values in (
            {"names": {1: "object"}},
            {"nc": 2},
            {"names": []},
            {"names": ["same", "same"]},
            {"names": {True: "object"}},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                inspect_calibration_data(self.write_yaml(**values))

    def test_yaml_does_not_allow_python_object_construction(self):
        self.source.write_text("!!python/object/apply:os.system ['false']\n")
        with self.assertRaisesRegex(ValueError, "calibration YAML is invalid"):
            inspect_calibration_data(self.source)

    def test_seeded_selection_is_repeatable_without_changing_global_rng(self):
        for index in range(30):
            (self.training / f"{index:02}.png").write_bytes(bytes([index]))
        data = inspect_calibration_data(self.write_yaml())
        config = get_preset("int8-512x768-raw")
        before = random.getstate()
        first = select_calibration_images(data, config)
        second = select_calibration_images(data, config)
        changed = select_calibration_images(data, replace(config, calibration_seed=7))
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len(first.images), round(32 * 0.3))
        self.assertEqual(len(set(first.images)), len(first.images))
        self.assertNotEqual(set(first.images), set(changed.images))
        self.assertNotEqual(set(first.images), set(data.images[: len(first.images)]))
        self.assertLessEqual(set(first.images), set(data.images))

    def test_random_sampling_can_select_every_position_and_historical_prefix_is_explicit(self):
        for index in range(10):
            (self.training / f"{index:02}.png").write_bytes(bytes([index]))
        data = inspect_calibration_data(self.write_yaml())
        config = replace(get_preset("int8-512x768-raw"), calibration_fraction=0.5)
        observed = {
            image
            for seed in range(40)
            for image in select_calibration_images(
                data, replace(config, calibration_seed=seed)
            ).images
        }
        self.assertEqual(observed, set(data.images))
        historical = select_calibration_images(
            data, replace(config, calibration_selection="sorted-first")
        )
        self.assertEqual(historical.images, data.images[:6])
        self.assertIsNone(historical.summary()["selection_seed"])

    def test_explicit_representative_list_fraction_one_hashes_exact_selected_content(self):
        listed = self.root / "representative.txt"
        listed.write_text("images/train/second.png\n")
        data = inspect_calibration_data(self.write_yaml(train="representative.txt"))
        selected = select_calibration_images(
            data, replace(get_preset("int8-512x768-raw"), calibration_fraction=1.0)
        )
        self.assertEqual(selected.images, (self.second,))
        self.assertEqual(
            selected.summary()["selected_images"],
            [
                {
                    "path": str(self.second),
                    "sha256": hashlib.sha256(self.second.read_bytes()).hexdigest(),
                }
            ],
        )
        self.first.write_bytes(b"unused image can change")
        selected.verify_contents()
        self.second.write_bytes(b"selected image changed")
        with self.assertRaisesRegex(ValueError, "content changed"):
            selected.verify_contents()
        updated = select_calibration_images(
            data, replace(get_preset("int8-512x768-raw"), calibration_fraction=1.0)
        )
        self.assertNotEqual(updated.summary(), selected.summary())


if __name__ == "__main__":
    unittest.main()
