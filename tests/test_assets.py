import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from axell_export.assets import load_negative_assets
from axell_export.augmentation import CopyPasteConfig


class NegativeAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.owned = self.root / "owned" / "train"
        self.owned.mkdir(parents=True)
        self.image_path = self.owned / "source.png"
        self.label_path = self.owned / "source.txt"
        self.manifest = self.root / "negative-assets.yaml"
        # Spatially distinct BGR values expose wrong channels and box geometry.
        rows, columns = np.indices((80, 100), dtype=np.uint8)
        self.image = np.stack((columns, rows, np.full_like(rows, 153)), axis=2)
        self.assertTrue(cv2.imwrite(str(self.image_path), self.image))
        self.label_path.write_text("0 0.25 0.25 0.5 0.5\n1 0.75 0.75 0.5 0.5\n")
        self.config = CopyPasteConfig(crop_inset=0)

    def write_manifest(self, **overrides):
        document = {
            "negative_class_ids": [0],
            "assets": [{"image": "owned/train/source.png", "label": "owned/train/source.txt"}],
        }
        document.update(overrides)
        self.manifest.write_text(yaml.safe_dump(document), encoding="utf-8")
        return self.manifest

    def test_selects_declared_source_class_and_keeps_bgr_pixel_geometry(self):
        crops = load_negative_assets(self.write_manifest(), self.config)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].dtype, np.uint8)
        np.testing.assert_array_equal(crops[0], self.image[:40, :50])
        self.assertTrue(crops[0].flags.c_contiguous)
        np.testing.assert_array_equal(crops[0][0, 0], [0, 0, 153])

    def test_inset_is_total_fraction_removed_from_both_sides(self):
        self.label_path.write_text("0 0.5 0.5 1 1\n")
        crops = load_negative_assets(self.write_manifest(), CopyPasteConfig(crop_inset=0.1))
        self.assertEqual(crops[0].shape, (72, 90, 3))
        np.testing.assert_array_equal(crops[0], self.image[4:76, 5:95])

    def test_asset_paths_resolve_beside_manifest_and_support_absolute_paths(self):
        source = self.write_manifest(
            assets=[{"image": str(self.image_path), "label": "owned/train/source.txt"}]
        )
        with patch("axell_export.assets.Path.cwd", return_value=self.root.parent):
            crops = load_negative_assets(source, self.config)
        np.testing.assert_array_equal(crops[0], self.image[:40, :50])

    def test_only_explicitly_listed_files_are_read(self):
        (self.owned / "unlisted.png").write_bytes(b"unreadable unlisted image")
        (self.owned / "unlisted.txt").write_bytes(b"invalid unlisted labels")
        source = self.write_manifest()
        with patch("axell_export.assets.cv2.imread", wraps=cv2.imread) as decode:
            crops = load_negative_assets(source, self.config)
        self.assertEqual(len(crops), 1)
        decode.assert_called_once_with(str(self.image_path), cv2.IMREAD_COLOR)

    def test_crops_are_independent_and_all_sources_remain_unchanged(self):
        source = self.write_manifest(negative_class_ids=[0, 1])
        files = sorted(self.root.rglob("*"))
        before = {
            path: hashlib.sha256(path.read_bytes()).digest() for path in files if path.is_file()
        }
        first, second = load_negative_assets(source, self.config)
        original_second = second.copy()
        first[:] = 0
        np.testing.assert_array_equal(second, original_second)
        after = {
            path: hashlib.sha256(path.read_bytes()).digest()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertEqual(files, sorted(self.root.rglob("*")))

    def test_near_boundary_rounding_tolerance_does_not_wrap_negative_indices(self):
        self.label_path.write_text("0 0.4999999 0.5000001 1 1\n")
        crops = load_negative_assets(self.write_manifest(), self.config)
        np.testing.assert_array_equal(crops[0], self.image)

    def test_rejects_empty_or_unexpected_manifest_fields(self):
        cases = (
            {"negative_class_ids": []},
            {"negative_class_ids": [True]},
            {"negative_class_ids": [-1]},
            {"negative_class_ids": [0.0]},
            {"negative_class_ids": [0, 0]},
            {"negative_class_ids": "0"},
            {"assets": []},
            {"assets": "owned/train"},
            {"assets": [{"image": "owned/train/source.png"}]},
            {
                "assets": [
                    {"image": "owned/train/source.png", "label": "owned/train/source.txt", "x": 1}
                ]
            },
            {"assets": [None]},
            {"download": "https://example.invalid/assets.zip"},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_negative_assets(self.write_manifest(**values), self.config)
        for raw in ("", "[]\n", "null\n", "negative_class_ids: [0]\n"):
            self.manifest.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_negative_assets(self.manifest, self.config)

    def test_rejects_duplicate_pairs_including_symlink_aliases(self):
        alias = self.owned / "alias.png"
        alias.symlink_to(self.image_path)
        entries = [
            {"image": "owned/train/source.png", "label": "owned/train/source.txt"},
            {"image": "owned/train/alias.png", "label": str(self.label_path)},
        ]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            load_negative_assets(self.write_manifest(assets=entries), self.config)

    def test_rejects_urls_directories_and_invalid_path_types(self):
        for value in (
            "https://example.invalid/a.png",
            "file:///a.png",
            "s3://bucket/a",
            "//host/a",
            "",
            1,
            None,
            True,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_negative_assets(
                    self.write_manifest(assets=[{"image": value, "label": str(self.label_path)}]),
                    self.config,
                )
        with self.assertRaises(FileNotFoundError):
            load_negative_assets(
                self.write_manifest(
                    assets=[{"image": str(self.owned), "label": str(self.label_path)}]
                ),
                self.config,
            )
        with self.assertRaisesRegex(ValueError, "local path"):
            load_negative_assets("https://example.invalid/manifest.yaml", self.config)

    def test_rejects_unsafe_yaml_duplicate_keys_and_invalid_utf8(self):
        for raw in (
            "!!python/object/apply:os.system ['false']\n",
            "negative_class_ids: [0]\nnegative_class_ids: [1]\nassets: []\n",
            "negative_class_ids: [0]\nassets: [\n",
        ):
            self.manifest.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_negative_assets(self.manifest, self.config)
        self.manifest.write_bytes(b"\xff\xfe")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_negative_assets(self.manifest, self.config)

    def test_rejects_malformed_labels_including_unselected_classes(self):
        invalid = (
            "0 0.5 0.5 0.5",
            "0 0.5 0.5 0.5 0.5 1",
            "-1 0.5 0.5 1 1",
            "1.0 0.5 0.5 1 1",
            "0 nan 0.5 1 1",
            "0 0.5 inf 1 1",
            "0 0.5 0.5 0 1",
            "0 0.5 0.5 -0.5 1",
            "0 0.1 0.5 0.5 1",
            "0 0.9 0.5 0.5 1",
            "0 0.5 0.5 1 2",
            "0 0.5 value 1 1",
            "99 0.5 0.5 1 2",
        )
        source = self.write_manifest()
        for text in invalid:
            self.label_path.write_text(text + "\n")
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "source.txt:1"):
                load_negative_assets(source, self.config)

    def test_rejects_unreadable_images_and_no_selected_boxes(self):
        source = self.write_manifest(negative_class_ids=[2])
        with self.assertRaisesRegex(ValueError, "no usable boxes"):
            load_negative_assets(source, self.config)
        source = self.write_manifest()
        self.image_path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            load_negative_assets(source, self.config)


if __name__ == "__main__":
    unittest.main()
