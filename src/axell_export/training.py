"""Connect negative-object Copy-Paste to the pinned YOLO detection trainer."""

from collections import defaultdict
from dataclasses import replace
from importlib import metadata
import os

from .augmentation import CopyPasteConfig, NegativeCopyPaste


if metadata.version("ultralytics") != "8.3.170":
    raise RuntimeError("Copy-Paste training integration requires ultralytics==8.3.170")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"

from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel  # noqa: E402


class CopyPasteDataset(YOLODataset):
    """Install the transform before workers start and retain it after Mosaic closes."""

    def __init__(self, *args, negative_assets, copy_paste_config=CopyPasteConfig(), **kwargs):
        if kwargs.get("task", "detect") != "detect":
            raise ValueError("negative-object Copy-Paste supports detection datasets only")
        self.copy_paste_config = copy_paste_config.validate()
        self.copy_paste_transform = NegativeCopyPaste(negative_assets, self.copy_paste_config)
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            # Format consumes Instances, so this transform must precede it.
            # The superclass builds a fresh sequence, including after close_mosaic().
            transforms.insert(-1, self.copy_paste_transform)
        return transforms

    def set_copy_paste_probability(self, probability):
        self.copy_paste_config = replace(self.copy_paste_config, probability=probability).validate()
        self.copy_paste_transform.config = self.copy_paste_config


def update_copy_paste_epoch(trainer):
    """Apply the epoch policy to workers, discarding batches prefetched with an old policy."""
    dataset = trainer.train_loader.dataset
    if not isinstance(dataset, CopyPasteDataset):
        raise TypeError("CopyPasteTrainer requires a CopyPasteDataset for training")
    stop_last = trainer.copy_paste_stop_last_epochs
    probability = trainer.copy_paste_config.probability
    if stop_last and trainer.epoch >= trainer.epochs - stop_last:
        probability = 0.0
    if dataset.copy_paste_config.probability == probability:
        return
    reset = getattr(trainer.train_loader, "reset", None)
    if not callable(reset):
        raise TypeError("Changing the epoch policy requires a resettable training loader")
    dataset.set_copy_paste_probability(probability)
    # InfiniteDataLoader.reset creates workers from the updated parent dataset.
    # Explicitly retire its old iterator so prefetched augmented batches cannot leak
    # into the disabled phase and old worker processes do not accumulate.
    iterator = getattr(trainer.train_loader, "iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    reset()


class CopyPasteTrainer(DetectionTrainer):
    """Single-process detection trainer using caller-supplied negative assets.

    Pass this class through ``functools.partial`` to ``YOLO.train(trainer=...)``.
    The standard training loop handles optimization, validation and saving. This
    adapter changes only training-data construction and the Copy-Paste schedule.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
        *,
        negative_assets,
        copy_paste_config=CopyPasteConfig(),
        copy_paste_stop_last_epochs=0,
    ):
        self.copy_paste_config = copy_paste_config.validate()
        if type(copy_paste_stop_last_epochs) is not int or copy_paste_stop_last_epochs < 0:
            raise ValueError("copy_paste_stop_last_epochs must be a nonnegative integer")
        self.copy_paste_stop_last_epochs = copy_paste_stop_last_epochs
        self.negative_assets = tuple(negative_assets)
        # Validate and own the input arrays before the superclass starts any setup.
        self.negative_assets = NegativeCopyPaste(
            self.negative_assets, self.copy_paste_config
        ).negative_assets
        resolved = get_cfg(cfg, overrides)
        device = resolved.device
        if isinstance(device, (list, tuple)) or (isinstance(device, str) and "," in device):
            raise ValueError("CopyPasteTrainer supports one process and one device only")
        if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("CopyPasteTrainer does not support distributed training")
        if resolved.task != "detect":
            raise ValueError("CopyPasteTrainer supports object detection only")
        if self.copy_paste_stop_last_epochs > (resolved.epochs or 100):
            raise ValueError("copy_paste_stop_last_epochs cannot exceed the training epochs")
        # YOLO passes its own callback mapping to the trainer. Keep this
        # trainer's additions private so later ordinary training is unaffected.
        callbacks = (
            defaultdict(list, {name: list(functions) for name, functions in _callbacks.items()})
            if _callbacks is not None
            else None
        )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=callbacks)
        self.callbacks["on_train_epoch_start"] = [
            callback
            for callback in self.callbacks["on_train_epoch_start"]
            if callback is not update_copy_paste_epoch
        ]
        self.add_callback("on_train_epoch_start", update_copy_paste_epoch)

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "val":
            return super().build_dataset(img_path, mode, batch)
        if mode != "train":
            raise ValueError("mode must be train or val")
        args = self.args
        if args.task != "detect":
            raise ValueError("CopyPasteTrainer supports object detection only")
        stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # These arguments follow the pinned build_yolo_dataset API. Dataset
        # construction finishes before DetectionTrainer creates its data loader.
        return CopyPasteDataset(
            img_path=img_path,
            imgsz=args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=args,
            rect=args.rect,
            cache=args.cache or None,
            single_cls=args.single_cls or False,
            stride=stride,
            pad=0.0,
            prefix="train: ",
            task=args.task,
            classes=args.classes,
            data=self.data,
            fraction=args.fraction,
            negative_assets=self.negative_assets,
            copy_paste_config=self.copy_paste_config,
        )
