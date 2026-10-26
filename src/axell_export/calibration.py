"""Validate a local training split and stage a calibration-only dataset description."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re

import yaml

from .config import ExportConfig


IMAGE_SUFFIXES = frozenset(
    {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
)
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


@dataclass(frozen=True)
class CalibrationData:
    """Local training image paths; images and annotations are never opened here."""

    source: Path
    source_sha256: str
    images: tuple[Path, ...]
    names: tuple[str, ...]

    def summary(self) -> dict:
        return {
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "split": "train",
            "image_count": len(self.images),
            "class_count": len(self.names),
            "training_paths_sha256": hashlib.sha256(
                "".join(f"{path}\n" for path in self.images).encode("utf-8")
            ).hexdigest(),
        }


def _image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationSelection:
    """Frozen ordered inputs and content identities for one calibration run."""

    data: CalibrationData
    images: tuple[Path, ...]
    sha256: tuple[str, ...]
    method: str
    seed: int
    fraction: float

    @property
    def names(self) -> tuple[str, ...]:
        return self.data.names

    def summary(self) -> dict:
        return {
            **self.data.summary(),
            "fraction": self.fraction,
            "selection_method": self.method,
            "selection_seed": self.seed if self.method == "seeded-random" else None,
            "effective_image_count": len(self.images),
            "selected_images": [
                {"path": str(path), "sha256": digest}
                for path, digest in zip(self.images, self.sha256, strict=True)
            ],
        }

    def verify_contents(self) -> None:
        for index in range(len(self.images)):
            self.verify_image(index)

    def verify_image(self, index: int) -> None:
        path = self.images[index]
        if _image_sha256(path) != self.sha256[index]:
            raise ValueError(f"calibration image content changed after selection: {path}")


def select_calibration_images(data: CalibrationData, config: ExportConfig) -> CalibrationSelection:
    """Sample training images without replacement and hash only selected files.

    The local RNG does not alter a caller's global random state. The historical
    sorted prefix remains available explicitly; seeded sampling does not itself
    guarantee class or scene balance. A curated train list with fraction=1 can
    instead define a caller-selected representative set.
    """
    config.validate()
    if config.precision != "int8":
        raise ValueError("calibration selection requires an INT8 configuration")
    ordered = tuple(sorted(data.images))
    count = round(len(ordered) * config.calibration_fraction)
    if count < 1:
        raise ValueError("calibration fraction selects zero images; increase fraction or inputs")
    if config.calibration_selection == "seeded-random":
        selected = tuple(random.Random(config.calibration_seed).sample(ordered, count))
    else:
        selected = ordered[:count]
    return CalibrationSelection(
        data=data,
        images=selected,
        sha256=tuple(_image_sha256(path) for path in selected),
        method=config.calibration_selection,
        seed=config.calibration_seed,
        fraction=config.calibration_fraction,
    )


def _local_path(value: object, parent: Path, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty local path")
    if _URI.match(value) or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be a local path without a URL or newline")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def _class_names(document: dict) -> tuple[str, ...]:
    names = document.get("names")
    if isinstance(names, dict):
        indexed = {}
        for key, value in names.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError("names keys must be consecutive class indices starting at zero")
            try:
                index = int(key)
            except ValueError as error:
                raise ValueError("names keys must be integer class indices") from error
            if str(index) != str(key) or index in indexed:
                raise ValueError("names keys must be unique integer class indices")
            indexed[index] = value
        if set(indexed) != set(range(len(indexed))):
            raise ValueError("names keys must be consecutive class indices starting at zero")
        names = [indexed[index] for index in range(len(indexed))]
    if not isinstance(names, list) or not names:
        raise ValueError("calibration YAML must provide a nonempty names list or index mapping")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("class names must be unique")
    count = document.get("nc", len(names))
    if isinstance(count, bool) or not isinstance(count, int) or count != len(names):
        raise ValueError("nc must match the number of class names")
    return tuple(names)


def _check_image(path: Path) -> Path:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported training image suffix: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"training image does not exist: {path}")
    if "\n" in str(path) or "\r" in str(path):
        raise ValueError("training image paths cannot contain newlines")
    return path.resolve()


def _images_from_entry(path: Path) -> list[Path]:
    if path.is_dir():
        images = [
            _check_image(child)
            for child in sorted(path.rglob("*"))
            if child.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            raise ValueError(f"training directory contains no supported images: {path}")
        return images
    if not path.is_file():
        raise FileNotFoundError(f"training split path does not exist: {path}")
    if path.suffix.lower() == ".txt":
        images = [
            _check_image(_local_path(line.strip(), path.parent, field="training image list entry"))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not images:
            raise ValueError(f"training image list is empty: {path}")
        return images
    return [_check_image(path)]


def inspect_calibration_data(source: str | Path) -> CalibrationData:
    """Read only YAML and optional text lists, selecting the explicit train split.

    A relative ``path`` is anchored at the YAML directory. Relative ``train``
    entries are anchored at ``path`` (or the YAML directory if omitted).
    Relative paths inside a text list are anchored at that list's directory.
    ``val`` and ``test`` never participate in calibration selection.
    """
    source = _local_path(str(source), Path.cwd(), field="calibration YAML")
    if source.suffix.lower() not in {".yaml", ".yml"} or not source.is_file():
        raise ValueError("calibration data must be an existing local .yaml or .yml file")
    source_bytes = source.read_bytes()
    try:
        document = yaml.safe_load(source_bytes.decode("utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("calibration YAML is invalid") from error
    if not isinstance(document, dict):
        raise ValueError("calibration YAML must be a mapping")
    if "download" in document:
        raise ValueError("download entries are not allowed in calibration YAML")
    names = _class_names(document)
    root = _local_path(document.get("path", "."), source.parent, field="dataset path")
    entries = document.get("train")
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        raise ValueError("calibration YAML must explicitly provide a nonempty train split")
    images = []
    for entry in entries:
        images.extend(_images_from_entry(_local_path(entry, root, field="train")))
    if len(set(images)) != len(images):
        raise ValueError("training image paths must be unique; duplicate paths were found")
    return CalibrationData(
        source=source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        images=tuple(sorted(images)),
        names=names,
    )


def write_calibration_data(
    data: CalibrationData | CalibrationSelection, run_dir: str | Path
) -> Path:
    """Write a verified train list and YAML inside a new private export run.

    The val alias exists only to satisfy the backend dataset schema. It points
    at the same training list and is not an independent validation dataset.
    """
    run_dir = Path(run_dir).resolve()
    image_list = run_dir / "calibration-train.txt"
    yaml_path = run_dir / "calibration.yaml"
    if image_list.exists() or yaml_path.exists():
        raise FileExistsError("calibration staging files already exist")
    with image_list.open("x", encoding="utf-8") as stream:
        stream.write("".join(f"{path}\n" for path in data.images))
    document = {
        "path": str(run_dir),
        "train": str(image_list),
        "val": str(image_list),
        "nc": len(data.names),
        "names": list(data.names),
    }
    with yaml_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
    return yaml_path
