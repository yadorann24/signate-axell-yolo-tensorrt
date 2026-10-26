"""Run a small real training loop on generated data, never competition inputs."""

from functools import partial
from importlib.util import find_spec
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    find_spec("torch") is not None and find_spec("ultralytics") is not None,
    "Training smoke test requires the runtime extra",
)
class TrainingSmokeTests(unittest.TestCase):
    def test_cpu_training_receives_pastes_then_stops_them_for_the_last_epoch(self):
        with tempfile.TemporaryDirectory(prefix="axell-training-smoke-") as temporary:
            root = Path(temporary)
            with patch.dict(
                os.environ,
                {
                    "YOLO_CONFIG_DIR": str(root / "settings"),
                    "YOLO_AUTOINSTALL": "false",
                    "YOLO_OFFLINE": "true",
                },
            ):
                import cv2
                import numpy as np
                import torch
                import yaml
                from ultralytics import YOLO
                from axell_export.augmentation import CopyPasteConfig
                from axell_export.training import CopyPasteTrainer

                old_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                self.addCleanup(torch.set_num_threads, old_threads)
                source = np.zeros((64, 64, 3), dtype=np.uint8)
                source[8:24, 8:24] = 80
                for split in ("train", "val"):
                    images = root / "images" / split
                    labels = root / "labels" / split
                    images.mkdir(parents=True)
                    labels.mkdir(parents=True)
                    for index in range(2):
                        self.assertTrue(cv2.imwrite(str(images / f"{index}.png"), source))
                        (labels / f"{index}.txt").write_text("0 0.25 0.25 0.25 0.25\n")
                data = root / "data.yaml"
                data.write_text(
                    yaml.safe_dump(
                        {
                            "path": str(root),
                            "train": "images/train",
                            "val": "images/val",
                            "names": ["synthetic-object"],
                        }
                    )
                )
                batches = []
                parameters = []

                class ObservedTrainer(CopyPasteTrainer):
                    def preprocess_batch(self, batch):
                        batches.append((self.epoch, batch["img"].clone(), batch["bboxes"].clone()))
                        return super().preprocess_batch(batch)

                def save_initial_parameter(trainer):
                    parameters.append(next(trainer.model.parameters()).detach().clone())

                def save_final_parameter(trainer):
                    parameters.append(next(trainer.model.parameters()).detach().clone())

                with (
                    patch("ultralytics.data.utils.check_font", return_value=None),
                    patch(
                        "ultralytics.utils.downloads.safe_download",
                        side_effect=AssertionError("No downloads"),
                    ),
                    patch(
                        "torch.hub.download_url_to_file",
                        side_effect=AssertionError("No weight downloads"),
                    ),
                ):
                    model = YOLO("yolo11n.yaml")
                    model.add_callback("on_train_start", save_initial_parameter)
                    model.add_callback("on_train_end", save_final_parameter)
                    model.train(
                        trainer=partial(
                            ObservedTrainer,
                            negative_assets=(np.full((12, 12, 3), 255, dtype=np.uint8),),
                            copy_paste_config=CopyPasteConfig(probability=1.0, max_pastes=1),
                            copy_paste_stop_last_epochs=1,
                        ),
                        data=str(data),
                        epochs=2,
                        batch=2,
                        nbs=2,
                        imgsz=64,
                        device="cpu",
                        workers=0,
                        project=str(root / "runs"),
                        name="smoke",
                        pretrained=False,
                        amp=False,
                        plots=False,
                        save=True,
                        val=True,
                        verbose=False,
                        optimizer="SGD",
                        lr0=0.01,
                        warmup_epochs=0.0,
                        close_mosaic=1,
                        mosaic=0.0,
                        mixup=0.0,
                        cutmix=0.0,
                        copy_paste=0.0,
                        degrees=0.0,
                        translate=0.0,
                        scale=0.0,
                        shear=0.0,
                        perspective=0.0,
                        flipud=0.0,
                        fliplr=0.0,
                        hsv_h=0.0,
                        hsv_s=0.0,
                        hsv_v=0.0,
                        bgr=0.0,
                    )
                self.assertEqual([epoch for epoch, _, _ in batches], [0, 1])
                self.assertTrue((batches[0][1] == 255).flatten(1).any(1).all().item())
                self.assertFalse((batches[1][1] == 255).any().item())
                for _, _, boxes in batches:
                    self.assertTrue(torch.allclose(boxes, torch.tensor([[0.25] * 4] * 2)))
                self.assertEqual(len(parameters), 2)
                self.assertFalse(torch.equal(parameters[0], parameters[1]))
                self.assertTrue(torch.isfinite(model.trainer.tloss).all().item())
                self.assertTrue((root / "runs" / "smoke" / "weights" / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
