"""Verify training integration with generated images and real worker processes."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    find_spec("torch") is not None and find_spec("ultralytics") is not None,
    "Training integration tests require the runtime extra",
)
class CopyPasteTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = tempfile.TemporaryDirectory(prefix="axell-training-settings-")
        cls.addClassCleanup(settings.cleanup)
        environment = patch.dict(
            os.environ,
            {
                "YOLO_CONFIG_DIR": settings.name,
                "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true",
            },
        )
        environment.start()
        cls.addClassCleanup(environment.stop)

        import numpy as np
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.data.augment import Format
        from ultralytics.data.build import InfiniteDataLoader, seed_worker
        from axell_export.augmentation import CopyPasteConfig, NegativeCopyPaste
        from axell_export.training import (
            CopyPasteDataset,
            CopyPasteTrainer,
            update_copy_paste_epoch,
        )

        cls.np = np
        cls.torch = torch
        cls.get_cfg = staticmethod(get_cfg)
        cls.Format = Format
        cls.InfiniteDataLoader = InfiniteDataLoader
        cls.seed_worker = staticmethod(seed_worker)
        cls.CopyPasteConfig = CopyPasteConfig
        cls.NegativeCopyPaste = NegativeCopyPaste
        cls.CopyPasteDataset = CopyPasteDataset
        cls.CopyPasteTrainer = CopyPasteTrainer
        cls.update_epoch = staticmethod(update_copy_paste_epoch)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)

    def setUp(self):
        import cv2

        temporary = tempfile.TemporaryDirectory(prefix="axell-training-images-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()
        source = self.np.zeros((128, 128, 3), dtype=self.np.uint8)
        source[8:24, 8:24] = (20, 30, 40)
        for index in range(4):
            self.assertTrue(cv2.imwrite(str(self.images / f"{index}.png"), source))
            (self.labels / f"{index}.txt").write_text(
                "0 0.125 0.125 0.125 0.125\n", encoding="utf-8"
            )
        self.input_hashes = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (self.images, self.labels)
            for path in folder.iterdir()
        }
        self.assets = (self.np.full((16, 12, 3), 255, dtype=self.np.uint8),)
        self.config = self.CopyPasteConfig(
            probability=1.0,
            max_pastes=1,
            max_attempts=100,
            crop_inset=0.0,
            area_mode="mean",
        )
        self.hyp = self.get_cfg()
        for name in (
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "bgr",
        ):
            setattr(self.hyp, name, 0.0)
        self.hyp.imgsz = 128
        self.hyp.task = "detect"
        self.hyp.cache = False
        self.hyp.rect = False
        self.hyp.workers = 2
        self.data = {"names": {0: "synthetic-object"}, "nc": 1, "channels": 3}
        random.seed(42)
        self.np.random.seed(42)

    def make_dataset(self, *, augment=True, config=None):
        return self.CopyPasteDataset(
            img_path=str(self.images),
            imgsz=128,
            batch_size=2,
            augment=augment,
            hyp=deepcopy(self.hyp),
            rect=False,
            cache=False,
            stride=32,
            pad=0.0,
            data=self.data,
            task="detect",
            negative_assets=self.assets,
            copy_paste_config=config or self.config,
        )

    def assert_inputs_unchanged(self):
        for name, digest in self.input_hashes.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest)
        self.np.testing.assert_array_equal(self.assets[0], 255)

    def assert_labels_equal(self, actual, baseline):
        self.assertTrue(self.torch.equal(actual["cls"], baseline["cls"]))
        self.assertTrue(self.torch.allclose(actual["bboxes"], baseline["bboxes"]))
        self.assertEqual(actual["bboxes"].shape[-1], 4)

    def test_real_dataset_changes_pixels_without_changing_positive_labels(self):
        baseline = self.make_dataset(config=replace(self.config, probability=0.0))[0]
        dataset = self.make_dataset()
        sample = dataset[0]
        self.assertEqual(tuple(sample["img"].shape), (3, 128, 128))
        self.assertTrue((sample["img"] == 255).any().item())
        self.assertFalse(self.torch.equal(sample["img"], baseline["img"]))
        self.assert_labels_equal(sample, baseline)
        self.assertTrue(
            self.torch.equal(sample["img"][:, 8:24, 8:24], baseline["img"][:, 8:24, 8:24])
        )
        transforms = dataset.transforms.transforms
        self.assertIsInstance(transforms[-2], self.NegativeCopyPaste)
        self.assertIsInstance(transforms[-1], self.Format)
        self.assert_inputs_unchanged()

    def test_validation_images_do_not_receive_negative_objects(self):
        dataset = self.make_dataset(augment=False)
        sample = dataset[0]
        self.assertFalse((sample["img"] == 255).any().item())
        self.assertEqual(len(sample["cls"]), 1)
        self.assertFalse(
            any(isinstance(item, self.NegativeCopyPaste) for item in dataset.transforms.transforms)
        )
        self.assert_inputs_unchanged()

    def test_trainer_factory_uses_custom_dataset_only_for_training(self):
        trainer = self.CopyPasteTrainer.__new__(self.CopyPasteTrainer)
        trainer.args = deepcopy(self.hyp)
        trainer.data = self.data
        trainer.model = None
        trainer.negative_assets = self.assets
        trainer.copy_paste_config = self.config
        training = trainer.build_dataset(str(self.images), mode="train", batch=2)
        validation = trainer.build_dataset(str(self.images), mode="val", batch=2)
        self.assertIsInstance(training, self.CopyPasteDataset)
        self.assertNotIsInstance(validation, self.CopyPasteDataset)
        self.assertTrue((training[0]["img"] == 255).any().item())
        self.assertFalse((validation[0]["img"] == 255).any().item())
        self.assertEqual(len(validation[0]["cls"]), 1)

    def test_trainer_registers_and_runs_the_epoch_stop_callback(self):
        def initialize_base(trainer, *args, **kwargs):
            trainer.callbacks = defaultdict(list)
            trainer.args = deepcopy(self.hyp)
            trainer.args.device = "cpu"
            trainer.epochs = 4

        with patch(
            "ultralytics.models.yolo.detect.train.DetectionTrainer.__init__",
            new=initialize_base,
        ):
            trainer = self.CopyPasteTrainer(
                negative_assets=self.assets,
                copy_paste_config=self.config,
                copy_paste_stop_last_epochs=2,
                overrides={"device": "cpu"},
            )
        trainer.train_loader = self.make_worker_loader(self.make_dataset())
        trainer.epochs = 4
        trainer.epoch = 2
        self.assertTrue(trainer.callbacks["on_train_epoch_start"])
        trainer.run_callbacks("on_train_epoch_start")
        for batch in trainer.train_loader:
            self.assertFalse((batch["img"] == 255).any().item())

    def test_custom_trainers_do_not_mutate_callbacks_shared_by_a_yolo_model(self):
        def existing_callback(trainer):
            return None

        shared = {
            "on_train_epoch_start": [existing_callback],
            "on_train_end": [existing_callback],
        }
        original = {event: list(callbacks) for event, callbacks in shared.items()}

        def initialize_base(trainer, cfg=None, overrides=None, _callbacks=None):
            # Ultralytics 8.3.170 intentionally retains this caller-owned mapping.
            trainer.callbacks = _callbacks
            trainer.args = deepcopy(self.hyp)
            trainer.epochs = 4

        with patch(
            "ultralytics.models.yolo.detect.train.DetectionTrainer.__init__",
            new=initialize_base,
        ):
            trainers = [
                self.CopyPasteTrainer(
                    negative_assets=self.assets,
                    copy_paste_config=self.config,
                    overrides={"device": "cpu"},
                    _callbacks=shared,
                )
                for _ in range(2)
            ]
        self.assertEqual(shared, original)
        for trainer in trainers:
            self.assertIsNot(trainer.callbacks, shared)
            self.assertEqual(trainer.callbacks["on_train_epoch_start"].count(self.update_epoch), 1)
            self.assertIn(existing_callback, trainer.callbacks["on_train_epoch_start"])
            for event in shared:
                self.assertIsNot(trainer.callbacks[event], shared[event])
        self.assertIsNot(
            trainers[0].callbacks["on_train_epoch_start"],
            trainers[1].callbacks["on_train_epoch_start"],
        )
        # A later ordinary training call must not execute the custom dataset hook.
        ordinary_trainer = SimpleNamespace()
        for callback in shared["on_train_epoch_start"]:
            callback(ordinary_trainer)

    def test_existing_epoch_hook_is_not_registered_twice(self):
        shared = {"on_train_epoch_start": [self.update_epoch, self.update_epoch]}

        def initialize_base(trainer, cfg=None, overrides=None, _callbacks=None):
            trainer.callbacks = _callbacks
            trainer.args = deepcopy(self.hyp)
            trainer.epochs = 4

        with patch(
            "ultralytics.models.yolo.detect.train.DetectionTrainer.__init__",
            new=initialize_base,
        ):
            trainer = self.CopyPasteTrainer(
                negative_assets=self.assets,
                _callbacks=shared,
                overrides={"device": "cpu"},
            )
        self.assertEqual(trainer.callbacks["on_train_epoch_start"], [self.update_epoch])
        self.assertEqual(shared["on_train_epoch_start"], [self.update_epoch] * 2)

    def test_multi_device_config_is_rejected_before_base_trainer_setup(self):
        configured = deepcopy(self.hyp)
        configured.device = "0,1"
        cases = (
            {"cfg": configured},
            {"overrides": {"device": "0,1"}},
            {"overrides": {"device": [0, 1]}},
        )
        for arguments in cases:
            with (
                self.subTest(arguments=arguments),
                patch(
                    "ultralytics.models.yolo.detect.train.DetectionTrainer.__init__"
                ) as initialize_base,
            ):
                with self.assertRaises(ValueError):
                    self.CopyPasteTrainer(negative_assets=self.assets, **arguments)
                initialize_base.assert_not_called()

    def test_distributed_process_is_rejected_before_base_trainer_setup(self):
        for rank in ("0", "1"):
            with (
                self.subTest(local_rank=rank),
                patch.dict(os.environ, {"LOCAL_RANK": rank}),
                patch(
                    "ultralytics.models.yolo.detect.train.DetectionTrainer.__init__"
                ) as initialize_base,
            ):
                with self.assertRaises(ValueError):
                    self.CopyPasteTrainer(negative_assets=self.assets, overrides={"device": "cpu"})
                initialize_base.assert_not_called()

    def test_closing_mosaic_rebuilds_exactly_one_active_copy_paste_transform(self):
        dataset = self.make_dataset()
        for _ in range(2):
            dataset.close_mosaic(deepcopy(self.hyp))
            count = sum(
                isinstance(item, self.NegativeCopyPaste) for item in dataset.transforms.transforms
            )
            self.assertEqual(count, 1)
            self.assertTrue((dataset[0]["img"] == 255).any().item())

    def test_copy_paste_stop_survives_mosaic_transform_rebuild(self):
        dataset = self.make_dataset()
        dataset.set_copy_paste_probability(0.0)
        dataset.close_mosaic(deepcopy(self.hyp))
        self.assertFalse((dataset[0]["img"] == 255).any().item())

    def make_worker_loader(self, dataset):
        generator = self.torch.Generator().manual_seed(44)
        loader = self.InfiniteDataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            collate_fn=dataset.collate_fn,
            worker_init_fn=self.seed_worker,
            generator=generator,
            persistent_workers=False,
            timeout=30,
        )
        self.addCleanup(loader.__del__)
        return loader

    def make_trainer(self, loader, *, epoch=0, epochs=4, stop_last_epochs=2):
        return SimpleNamespace(
            train_loader=loader,
            epoch=epoch,
            epochs=epochs,
            copy_paste_config=self.config,
            copy_paste_stop_last_epochs=stop_last_epochs,
        )

    def test_worker_batches_stop_pasting_at_epoch_boundary_without_stale_prefetch(self):
        dataset = self.make_dataset()
        loader = self.make_worker_loader(dataset)
        trainer = self.make_trainer(loader)
        expected_boxes = self.torch.tensor([[0.125] * 4] * 2)
        with patch.object(loader, "reset", wraps=loader.reset) as reset:
            self.update_epoch(trainer)
            for batch in loader:
                self.assertTrue((batch["img"] == 255).flatten(1).any(1).all().item())
                self.assertTrue(self.torch.allclose(batch["bboxes"], expected_boxes))
                self.assertEqual(batch["cls"].tolist(), [[0.0], [0.0]])
            reset.assert_not_called()

            trainer.epoch = 2
            self.update_epoch(trainer)
            reset.assert_called_once()
            for batch in loader:
                self.assertFalse((batch["img"] == 255).any().item())
                self.assertTrue(self.torch.allclose(batch["bboxes"], expected_boxes))
                self.assertEqual(batch["cls"].tolist(), [[0.0], [0.0]])

            trainer.epoch = 3
            self.update_epoch(trainer)
            reset.assert_called_once()
        self.assert_inputs_unchanged()

    def test_resumed_final_epoch_discards_already_prefetched_augmented_images(self):
        loader = self.make_worker_loader(self.make_dataset())
        trainer = self.make_trainer(loader, epoch=3)
        self.update_epoch(trainer)
        for batch in loader:
            self.assertFalse((batch["img"] == 255).any().item())
            self.assertEqual(len(batch["cls"]), 2)

    def test_zero_stop_epochs_keeps_augmentation_through_final_epoch(self):
        loader = self.make_worker_loader(self.make_dataset())
        trainer = self.make_trainer(loader, epoch=3, stop_last_epochs=0)
        with patch.object(loader, "reset", wraps=loader.reset) as reset:
            self.update_epoch(trainer)
            reset.assert_not_called()
        for batch in loader:
            self.assertTrue((batch["img"] == 255).flatten(1).any(1).all().item())


if __name__ == "__main__":
    unittest.main()
