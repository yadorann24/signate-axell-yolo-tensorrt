"""Add explicitly selected background distractors without changing target labels.

Input assets are already cropped BGR images, normally prepared by the asset
catalogue. This transform runs before Ultralytics' ``Format`` transform, when
images are NumPy arrays and boxes are still held by ``Instances``.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
import random

import cv2
import numpy as np


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class CopyPasteConfig:
    probability: float = 0.5
    max_pastes: int = 4
    max_attempts: int = 50
    crop_inset: float = 0.1
    area_mode: str = "mean"
    area_jitter: tuple[float, float] = (0.7, 1.3)
    area_fraction: tuple[float, float] = (0.001, 0.10)

    def __post_init__(self):
        self.validate()
        # JSON arrays become immutable pairs as well.
        object.__setattr__(self, "area_jitter", tuple(self.area_jitter))
        object.__setattr__(self, "area_fraction", tuple(self.area_fraction))

    def validate(self) -> "CopyPasteConfig":
        if not 0 <= _number(self.probability, "probability") <= 1:
            raise ValueError("probability must be in [0, 1]")
        if not 0 <= _number(self.crop_inset, "crop_inset") < 1:
            raise ValueError("crop_inset must be in [0, 1)")
        for name in ("max_pastes", "max_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.area_mode not in ("mean", "jitter"):
            raise ValueError("area_mode must be mean or jitter")
        for name in ("area_jitter", "area_fraction"):
            pair = getattr(self, name)
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"{name} must contain lower and upper bounds")
            lower, upper = (_number(value, name) for value in pair)
            if not 0 < lower <= upper:
                raise ValueError(f"{name} must have 0 < lower <= upper")
            if name == "area_fraction" and upper > 1:
                raise ValueError("area_fraction upper bound must be at most 1")
        return self


def _validate_image(image, name: str) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError(f"{name} must be a nonempty HWC BGR uint8 image")


def _visible_positive_boxes(labels: dict, width: int, height: int) -> np.ndarray:
    from ultralytics.utils.instance import Instances

    instances = labels.get("instances")
    if not isinstance(instances, Instances):
        raise ValueError("labels.instances must be an Ultralytics Instances object")
    if not isinstance(instances.normalized, (bool, np.bool_)):
        raise ValueError("instances.normalized must be a boolean")
    classes = labels.get("cls")
    boxes = instances.bboxes
    if (
        not isinstance(boxes, np.ndarray)
        or boxes.ndim != 2
        or boxes.shape[1] != 4
        or boxes.dtype.kind not in "fiu"
        or not np.isfinite(boxes).all()
    ):
        raise ValueError("instances must contain finite numeric boxes shaped (N, 4)")
    if (
        not isinstance(classes, np.ndarray)
        or classes.shape != (len(boxes), 1)
        or classes.dtype.kind not in "fiu"
        or not np.isfinite(classes).all()
        or np.any(classes < 0)
        or np.any(classes != np.floor(classes))
    ):
        raise ValueError("labels.cls must contain nonnegative class IDs shaped (N, 1)")

    # Conversion and denormalization must never mutate labels held in RAM caches.
    geometry = deepcopy(instances)
    geometry._bboxes.bboxes = boxes.astype(np.float64, copy=True)
    # Only box geometry is needed. Segments can legitimately be None in callers.
    geometry.segments = np.empty((0, 0, 2), dtype=np.float64)
    geometry.keypoints = None
    geometry.convert_bbox("xyxy")
    geometry.denormalize(width, height)
    absolute = geometry.bboxes
    if not np.isfinite(absolute).all() or np.any(absolute[:, 2:] <= absolute[:, :2]):
        raise ValueError("positive bounding boxes must have finite positive area")
    absolute[:, [0, 2]] = np.clip(absolute[:, [0, 2]], 0, width)
    absolute[:, [1, 3]] = np.clip(absolute[:, [1, 3]], 0, height)
    visible = np.all(absolute[:, 2:] > absolute[:, :2], axis=1)
    return absolute[visible]


def _overlaps(candidate: tuple[int, int, int, int], occupied: list) -> bool:
    x1, y1, x2, y2 = candidate
    return any(
        x1 < other[2] and x2 > other[0] and y1 < other[3] and y2 > other[1] for other in occupied
    )


class NegativeCopyPaste:
    """Paste background-only rectangles while preserving every positive box.

    Assets must not contain any of the detector's target classes. Pasted assets
    deliberately receive no class labels. Python's global RNG follows the seed
    set by the training framework in each data-loader worker.
    """

    def __init__(
        self,
        negative_assets: tuple[np.ndarray, ...] | list[np.ndarray],
        config: CopyPasteConfig = CopyPasteConfig(),
    ):
        if not isinstance(config, CopyPasteConfig):
            raise ValueError("config must be a CopyPasteConfig")
        self.config = config.validate()
        if not isinstance(negative_assets, (tuple, list)) or not negative_assets:
            raise ValueError("negative_assets must be a nonempty list or tuple")
        copied = []
        for index, asset in enumerate(negative_assets):
            _validate_image(asset, f"negative_assets[{index}]")
            immutable = np.ascontiguousarray(asset).copy()
            immutable.flags.writeable = False
            copied.append(immutable)
        self.negative_assets = tuple(copied)

    def __call__(self, labels: dict) -> dict:
        if not isinstance(labels, dict):
            raise ValueError("labels must be a dictionary")
        image = labels.get("img")
        _validate_image(image, "labels.img")
        height, width = image.shape[:2]
        boxes = _visible_positive_boxes(labels, width, height)
        result = dict(labels)
        result["img"] = image.copy()
        if not len(boxes) or self.config.probability == 0:
            return result
        if random.random() >= self.config.probability:
            return result

        occupied = list(boxes)
        areas = list(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1))
        initial_lower, initial_upper = np.percentile(areas, (20, 80))
        # Keep the original discrete size distribution. With two differently
        # sized boxes neither may lie between the interpolated quantiles.
        jitter_candidates = tuple(
            area for area in areas if initial_lower <= area <= initial_upper
        ) or tuple(areas)
        for _ in range(random.randint(1, self.config.max_pastes)):
            asset = random.choice(self.negative_assets)
            if self.config.area_mode == "mean":
                target_area = float(np.mean(areas))
            else:
                target_area = random.choice(jitter_candidates)
                target_area *= random.uniform(*self.config.area_jitter)
                target_area = float(
                    np.clip(
                        target_area,
                        self.config.area_fraction[0] * height * width,
                        self.config.area_fraction[1] * height * width,
                    )
                )
            aspect_ratio = asset.shape[1] / asset.shape[0]
            paste_width = max(1, round(math.sqrt(target_area * aspect_ratio)))
            paste_height = max(1, round(math.sqrt(target_area / aspect_ratio)))
            if paste_width > width or paste_height > height:
                continue

            for _ in range(self.config.max_attempts):
                x1 = random.randint(0, width - paste_width)
                y1 = random.randint(0, height - paste_height)
                candidate = (x1, y1, x1 + paste_width, y1 + paste_height)
                if _overlaps(candidate, occupied):
                    continue
                patch = cv2.resize(asset, (paste_width, paste_height))
                result["img"][y1 : y1 + paste_height, x1 : x1 + paste_width] = patch
                occupied.append(candidate)
                areas.append(paste_width * paste_height)
                break
        return result
