"""Load explicitly listed, user-owned negative crops without changing source files.

Class IDs belong to the source asset labels, not the competition label mapping.
The caller must verify that selected classes exclude all target brands and that
the images may be used for the intended purpose. This loader cannot establish
image rights or infer whether a visual object belongs to a target brand.
"""

from math import ceil, floor, isfinite
from pathlib import Path
import re
from typing import TYPE_CHECKING

import cv2
import numpy as np
import yaml

if TYPE_CHECKING:
    from .augmentation import CopyPasteConfig


_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_CLASS_ID = re.compile(r"[0-9]+")
_BOX_TOLERANCE = 1e-6


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects ambiguous duplicate mapping keys."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in result:
                    raise ValueError(f"negative asset manifest contains duplicate key: {key!r}")
                result[key] = self.construct_object(value_node, deep=deep)
            except TypeError as error:
                raise ValueError("negative asset manifest keys must be strings") from error
        return result


def _local_file(value: object, parent: Path, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty local file path")
    if (
        _SCHEME.match(value)
        or value.startswith(("//", "\\\\"))
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise ValueError(f"{field} must be a local path without a URL or control characters")
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist or is not a file: {path}")
    return path


def _manifest(source: Path) -> tuple[frozenset[int], tuple[tuple[Path, Path], ...]]:
    try:
        document = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        raise ValueError("negative asset manifest must contain valid, safe UTF-8 YAML") from error
    if not isinstance(document, dict) or set(document) != {"negative_class_ids", "assets"}:
        raise ValueError("negative asset manifest must contain only negative_class_ids and assets")
    class_ids = document["negative_class_ids"]
    if not isinstance(class_ids, list) or not class_ids:
        raise ValueError("negative_class_ids must be a nonempty list of nonnegative integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in class_ids
    ):
        raise ValueError("negative_class_ids must contain only nonnegative integers")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("negative_class_ids must not contain duplicates")
    entries = document["assets"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("assets must be a nonempty list of explicit image/label pairs")
    pairs = []
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"image", "label"}:
            raise ValueError(f"assets[{index}] must contain only image and label")
        pair = (
            _local_file(entry["image"], source.parent, field=f"assets[{index}].image"),
            _local_file(entry["label"], source.parent, field=f"assets[{index}].label"),
        )
        if pair in seen:
            raise ValueError(f"assets[{index}] duplicates an image/label pair")
        seen.add(pair)
        pairs.append(pair)
    return frozenset(class_ids), tuple(pairs)


def _boxes(label: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    try:
        lines = label.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"asset label must be UTF-8 text: {label}") from error
    boxes = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"{label}:{line_number}"
        fields = line.split()
        if len(fields) != 5 or not _CLASS_ID.fullmatch(fields[0]):
            raise ValueError(f"{location}: expected YOLO class_id cx cy width height")
        class_id = int(fields[0])
        try:
            center_x, center_y, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{location}: bounding box coordinates must be numeric") from error
        if not all(isfinite(value) for value in (center_x, center_y, width, height)):
            raise ValueError(f"{location}: bounding box coordinates must be finite")
        if width <= 0 or height <= 0:
            raise ValueError(f"{location}: bounding box width and height must be positive")
        left, top = center_x - width / 2, center_y - height / 2
        right, bottom = center_x + width / 2, center_y + height / 2
        if (
            left < -_BOX_TOLERANCE
            or top < -_BOX_TOLERANCE
            or right > 1 + _BOX_TOLERANCE
            or bottom > 1 + _BOX_TOLERANCE
        ):
            raise ValueError(
                f"{location}: bounding box must lie within the normalized image bounds"
            )
        boxes.append((class_id, (max(0.0, left), max(0.0, top), min(1.0, right), min(1.0, bottom))))
    return boxes


def _crop(image: np.ndarray, box: tuple[float, float, float, float], inset: float) -> np.ndarray:
    left, top, right, bottom = box
    horizontal_inset = (right - left) * inset / 2
    vertical_inset = (bottom - top) * inset / 2
    height, width = image.shape[:2]
    # Include pixels touched by the inset box; the epsilon avoids rounding a
    # mathematically integral floating-point boundary to its adjacent pixel.
    x1 = max(0, floor((left + horizontal_inset) * width + 1e-9))
    y1 = max(0, floor((top + vertical_inset) * height + 1e-9))
    x2 = min(width, ceil((right - horizontal_inset) * width - 1e-9))
    y2 = min(height, ceil((bottom - vertical_inset) * height - 1e-9))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("selected negative bounding box has no pixels after cropping")
    return image[y1:y2, x1:x2].copy()


def load_negative_assets(
    manifest_path: str | Path, config: "CopyPasteConfig"
) -> tuple[np.ndarray, ...]:
    """Return independent BGR uint8 crops in manifest and annotation order.

    The YAML schema is ``{negative_class_ids: [0, 2], assets: [{image:
    'owned/train/a.png', label: 'owned/train/a.txt'}]}``. Relative paths resolve
    beside this manifest. All annotations must be valid YOLO detection rows;
    only explicitly selected source classes produce crops. ``crop_inset`` is
    the total fraction removed per axis (0.1 removes 5% from each side).

    Images are decoded without label caches, repairs, directory discovery, or
    downloads. Source images, labels, and the manifest are never written.
    """
    config.validate()
    if not isinstance(manifest_path, (str, Path)):
        raise ValueError("negative asset manifest must be a local file path")
    source = _local_file(str(manifest_path), Path.cwd(), field="negative asset manifest")
    class_ids, pairs = _manifest(source)
    crops = []
    for image_path, label_path in pairs:
        boxes = _boxes(label_path)
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        except cv2.error as error:
            raise ValueError(f"negative asset image could not be decoded: {image_path}") from error
        if image is None or image.size == 0:
            raise ValueError(f"negative asset image could not be decoded: {image_path}")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"negative asset image must decode to BGR uint8: {image_path}")
        for class_id, box in boxes:
            if class_id in class_ids:
                crops.append(_crop(image, box, config.crop_inset))
    if not crops:
        raise ValueError("negative asset manifest has no usable boxes for negative_class_ids")
    return tuple(crops)
