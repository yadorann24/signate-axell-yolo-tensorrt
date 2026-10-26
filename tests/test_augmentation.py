"""Synthetic-image regression checks for background-only Copy-Paste."""

from copy import deepcopy
from importlib.util import find_spec
import random
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    find_spec("ultralytics") is not None,
    "Augmentation tests require the runtime extra",
)
class NegativeCopyPasteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from ultralytics.utils.instance import Instances

        from axell_export.augmentation import CopyPasteConfig, NegativeCopyPaste

        cls.np = np
        cls.Instances = Instances
        cls.CopyPasteConfig = CopyPasteConfig
        cls.NegativeCopyPaste = NegativeCopyPaste

    def setUp(self):
        previous_state = random.getstate()
        self.addCleanup(random.setstate, previous_state)
        random.seed(31)

    def labels(self, boxes=None, normalized=False, bbox_format="xyxy"):
        if boxes is None:
            boxes = [[0, 0, 10, 10]]
        boxes = self.np.asarray(boxes, dtype=self.np.float32).reshape(-1, 4)
        return {
            "img": self.np.zeros((100, 100, 3), dtype=self.np.uint8),
            "cls": self.np.zeros((len(boxes), 1), dtype=self.np.float32),
            "instances": self.Instances(
                boxes,
                segments=None,
                normalized=normalized,
                bbox_format=bbox_format,
            ),
            "im_file": "synthetic-only.png",
        }

    def transform(self, **overrides):
        values = {"probability": 1, "max_pastes": 1}
        values.update(overrides)
        asset = self.np.full((10, 10, 3), 200, dtype=self.np.uint8)
        return self.NegativeCopyPaste([asset], self.CopyPasteConfig(**values))

    def test_probability_one_pastes_without_mutating_cached_image_or_labels(self):
        labels = self.labels()
        image_before = labels["img"].copy()
        boxes_before = labels["instances"].bboxes.copy()
        classes_before = labels["cls"].copy()
        result = self.transform()(labels)
        self.assertGreater(self.np.count_nonzero(result["img"]), 0)
        self.assertIsNot(result, labels)
        self.assertIsNot(result["img"], labels["img"])
        self.assertIs(result["instances"], labels["instances"])
        self.assertIs(result["cls"], labels["cls"])
        self.assertEqual(set(result), set(labels))
        self.np.testing.assert_array_equal(labels["img"], image_before)
        self.np.testing.assert_array_equal(labels["instances"].bboxes, boxes_before)
        self.np.testing.assert_array_equal(labels["cls"], classes_before)
        self.np.testing.assert_array_equal(result["img"][:10, :10], image_before[:10, :10])

    def test_normalized_xywh_is_preserved_and_positive_pixels_are_untouched(self):
        labels = self.labels([[0.25, 0.25, 0.5, 0.5]], True, "xywh")
        original = labels["instances"].bboxes.copy()
        transform = self.transform()
        with patch("axell_export.augmentation.random.randint", side_effect=[1, 0, 0, 50, 50]):
            result = transform(labels)
        self.assertFalse(self.np.any(result["img"][:50, :50]))
        self.assertTrue(self.np.all(result["img"][50:, 50:] == 200))
        self.assertTrue(labels["instances"].normalized)
        self.assertEqual(labels["instances"]._bboxes.format, "xywh")
        self.np.testing.assert_array_equal(labels["instances"].bboxes, original)

    def test_probability_zero_and_empty_gt_leave_image_identical(self):
        for transform, labels in (
            (self.transform(probability=0), self.labels()),
            (self.transform(), self.labels([])),
        ):
            with patch("axell_export.augmentation.random.random") as random_call:
                result = transform(labels)
            random_call.assert_not_called()
            self.np.testing.assert_array_equal(result["img"], labels["img"])
            self.assertFalse(self.np.shares_memory(result["img"], labels["img"]))

    def test_probability_gate_can_skip_a_nonempty_image(self):
        transform = self.transform(probability=0.5)
        labels = self.labels()
        with patch("axell_export.augmentation.random.random", return_value=0.8):
            result = transform(labels)
        self.np.testing.assert_array_equal(result["img"], labels["img"])

    def test_new_pastes_cannot_overlap_previous_pastes(self):
        transform = self.transform(max_pastes=3, max_attempts=2)
        with patch(
            "axell_export.augmentation.random.randint",
            side_effect=[3, 20, 20, 20, 20, 40, 40, 60, 60],
        ):
            result = transform(self.labels())
        self.assertEqual(self.np.count_nonzero(result["img"][:, :, 0]), 300)
        for start in (20, 40, 60):
            self.assertTrue(
                self.np.all(result["img"][start : start + 10, start : start + 10] == 200)
            )

    def test_attempt_limit_stops_when_positive_object_fills_image(self):
        transform = self.transform(max_attempts=3)
        labels = self.labels([[0, 0, 100, 100]])
        with patch(
            "axell_export.augmentation.random.randint", side_effect=[1, 0, 0, 0, 0, 0, 0]
        ) as draw:
            result = transform(labels)
        self.assertEqual(draw.call_count, 7)
        self.np.testing.assert_array_equal(result["img"], labels["img"])

    def test_partially_outside_boxes_protect_visible_positive_pixels(self):
        transform = self.transform()
        labels = self.labels([[-10, -10, 20, 20]])
        with patch("axell_export.augmentation.random.randint", side_effect=[1, 0, 0, 40, 40]):
            result = transform(labels)
        self.assertFalse(self.np.any(result["img"][:20, :20]))
        self.assertTrue(self.np.all(result["img"][40:60, 40:60] == 200))
        self.np.testing.assert_array_equal(labels["instances"].bboxes, [[-10, -10, 20, 20]])

    def test_fully_outside_boxes_are_not_used_to_estimate_paste_size(self):
        labels = self.labels([[110, 110, 120, 120]])
        result = self.transform()(labels)
        self.np.testing.assert_array_equal(result["img"], labels["img"])

    def test_jitter_area_is_bounded_as_fraction_of_image(self):
        transform = self.transform(
            area_mode="jitter", area_jitter=(10, 10), area_fraction=(0.04, 0.04)
        )
        result = transform(self.labels())
        self.assertEqual(self.np.count_nonzero(result["img"][:, :, 0]), 400)

    def test_jitter_falls_back_to_discrete_gt_areas_when_quantile_band_is_empty(self):
        # P20=41.6 and P80=118.4 contain neither actual area (16, 144).
        for selection, expected_area in ((0, 16), (-1, 144)):
            with self.subTest(expected_area=expected_area):
                sampled_pools = []

                def choose(pool):
                    if isinstance(pool[0], self.np.ndarray):
                        return pool[0]
                    sampled_pools.append(tuple(pool))
                    return pool[selection]

                transform = self.transform(area_mode="jitter", area_jitter=(1, 1))
                labels = self.labels([[0, 0, 4, 4], [10, 0, 22, 12]])
                with (
                    patch("axell_export.augmentation.random.choice", side_effect=choose),
                    patch("axell_export.augmentation.random.randint", side_effect=[1, 30, 30]),
                ):
                    result = transform(labels)
                self.assertEqual(sampled_pools, [(16, 144)])
                self.assertEqual(self.np.count_nonzero(result["img"][:, :, 0]), expected_area)

    def test_jitter_uses_only_initial_gt_areas_inside_quantile_band(self):
        sampled_pools = []

        def choose(pool):
            if isinstance(pool[0], self.np.ndarray):
                return pool[0]
            sampled_pools.append(tuple(pool))
            return pool[-1]

        labels = self.labels(
            [[0, 0, 2, 2], [15, 0, 19, 4], [30, 0, 36, 6], [45, 0, 53, 8], [60, 0, 70, 10]]
        )
        transform = self.transform(area_mode="jitter", area_jitter=(0.5, 0.5), max_pastes=3)
        with (
            patch("axell_export.augmentation.random.choice", side_effect=choose),
            patch(
                "axell_export.augmentation.random.randint", side_effect=[3, 20, 20, 40, 40, 60, 60]
            ),
        ):
            result = transform(labels)
        # Pasted rectangles have area 36 after rounding. They must not alter
        # the initial candidate distribution or introduce duplicate candidates.
        self.assertEqual(sampled_pools, [(16, 36, 64)] * 3)
        self.assertEqual(self.np.count_nonzero(result["img"][:, :, 0]), 3 * 36)

    def test_fixed_seed_repeats_same_augmentation(self):
        transform = self.transform(max_pastes=4, area_mode="jitter")
        random.seed(201)
        first = transform(self.labels())
        random.seed(201)
        second = transform(self.labels())
        self.np.testing.assert_array_equal(first["img"], second["img"])

    def test_assets_are_copied_and_preserve_bgr_channel_values(self):
        asset = self.np.zeros((10, 10, 3), dtype=self.np.uint8)
        asset[:] = (10, 20, 30)
        transform = self.NegativeCopyPaste(
            [asset], self.CopyPasteConfig(probability=1, max_pastes=1)
        )
        asset[:] = 0
        self.assertFalse(transform.negative_assets[0].flags.writeable)
        result = transform(self.labels())
        pasted = result["img"][self.np.any(result["img"] != 0, axis=2)]
        self.assertGreater(len(pasted), 0)
        self.np.testing.assert_array_equal(pasted, self.np.tile([10, 20, 30], (len(pasted), 1)))

    def test_bad_assets_are_rejected(self):
        for assets in (
            [],
            [None],
            [self.np.zeros((5, 5), dtype=self.np.uint8)],
            [self.np.zeros((5, 5, 3))],
        ):
            with self.subTest(assets=str(type(assets[0])) if assets else "empty"):
                with self.assertRaises(ValueError):
                    self.NegativeCopyPaste(assets)

    def test_invalid_labels_and_boxes_are_rejected(self):
        malformed = []
        labels = self.labels()
        labels["cls"] = self.np.array([0])
        malformed.append(labels)
        labels = self.labels()
        labels["cls"][0, 0] = 0.5
        malformed.append(labels)
        labels = self.labels()
        labels["img"] = labels["img"].astype(float)
        malformed.append(labels)
        labels = self.labels()
        labels["instances"] = None
        malformed.append(labels)
        labels = self.labels()
        labels["instances"].normalized = "False"
        malformed.append(labels)
        malformed.extend(
            self.labels([box]) for box in ([0, 0, 0, 5], [5, 0, 2, 5], [0, 0, float("nan"), 5])
        )
        for labels in malformed:
            with self.subTest(labels=type(labels["instances"])):
                with self.assertRaises(ValueError):
                    self.transform()(labels)

    def test_original_segments_and_keypoints_are_preserved(self):
        labels = self.labels([[0.05, 0.05, 0.1, 0.1]], True, "xywh")
        labels["instances"].segments = self.np.array([[[0.0, 0.0], [0.1, 0.1]]])
        labels["instances"].keypoints = self.np.array([[[0.05, 0.05, 1]]])
        before = deepcopy(labels["instances"])
        self.transform()(labels)
        self.np.testing.assert_array_equal(labels["instances"].segments, before.segments)
        self.np.testing.assert_array_equal(labels["instances"].keypoints, before.keypoints)

    def test_invalid_config_is_rejected_at_construction(self):
        invalid = (
            {"probability": -0.1},
            {"probability": 1.1},
            {"probability": True},
            {"probability": float("nan")},
            {"max_pastes": 0},
            {"max_pastes": True},
            {"max_attempts": 0},
            {"max_attempts": 3.5},
            {"crop_inset": -0.1},
            {"crop_inset": 1},
            {"area_mode": "unknown"},
            {"area_jitter": (1, 0.5)},
            {"area_jitter": (0, 1)},
            {"area_jitter": (1, float("inf"))},
            {"area_fraction": (0.1, 1.1)},
            {"area_fraction": [0.1]},
            {"area_jitter": "0.7,1.3"},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.CopyPasteConfig(**values)

    def test_json_pairs_are_immutable(self):
        values = [0.5, 1.5]
        config = self.CopyPasteConfig(area_jitter=values)
        values[0] = 99
        self.assertEqual(config.area_jitter, (0.5, 1.5))
        with self.assertRaises(AttributeError):
            config.probability = 0


if __name__ == "__main__":
    unittest.main()
