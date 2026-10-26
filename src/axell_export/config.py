"""Explicit export presets; a precision request is not a hardware benchmark."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ExportConfig:
    preset: str
    image_size: tuple[int, int]
    precision: str
    nms: bool
    workspace_gb: float
    simplify: bool = True
    batch_size: int = 1
    dynamic: bool = False
    calibration_fraction: float = 0.30
    calibration_selection: str = "seeded-random"
    calibration_seed: int = 42
    # Effective NMS defaults in the pinned Ultralytics 8.3.170 exporter.
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300

    def validate(self) -> "ExportConfig":
        if not isinstance(self.preset, str) or not self.preset.strip():
            raise ValueError("preset must be a nonempty name")
        if not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2:
            raise ValueError("image_size must contain height and width")
        if any(type(v) is not int or v <= 0 or v % 32 for v in self.image_size):
            raise ValueError("height and width must be positive multiples of 32")
        if self.precision not in {"fp16", "int8"}:
            raise ValueError("precision must be fp16 or int8")
        for name in ("nms", "simplify", "dynamic"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("These export profiles support batch_size=1")
        if self.dynamic:
            raise ValueError("These export profiles require dynamic=False")
        for name in ("workspace_gb", "calibration_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.calibration_fraction > 1:
            raise ValueError("calibration_fraction must be in (0, 1]")
        if not isinstance(self.calibration_selection, str) or self.calibration_selection not in {
            "seeded-random",
            "sorted-first",
        }:
            raise ValueError("calibration_selection must be seeded-random or sorted-first")
        if type(self.calibration_seed) is not int or self.calibration_seed < 0:
            raise ValueError("calibration_seed must be a nonnegative integer")
        for name in ("conf", "iou"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        # The pinned exporter replaces conf=0 with 0.25; never silently change it.
        if self.conf == 0:
            raise ValueError("conf must be greater than zero for the pinned NMS exporter")
        if type(self.max_det) is not int or self.max_det < 1:
            raise ValueError("max_det must be a positive integer")
        if not self.nms and (self.conf, self.iou, self.max_det) != (0.25, 0.7, 300):
            raise ValueError("conf, iou, and max_det overrides require an NMS preset")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def export_kwargs(self) -> dict:
        self.validate()
        kwargs = {
            "format": "engine",
            "imgsz": tuple(self.image_size),
            "half": self.precision == "fp16",
            "int8": self.precision == "int8",
            "nms": self.nms,
            "workspace": self.workspace_gb,
            "simplify": self.simplify,
            "batch": self.batch_size,
            "dynamic": self.dynamic,
        }
        if self.precision == "int8":
            kwargs["fraction"] = self.calibration_fraction
        if self.nms:
            kwargs.update(conf=self.conf, iou=self.iou, max_det=self.max_det)
        return kwargs


_PRESETS = (
    ExportConfig("fp16-640-raw-w2", (640, 640), "fp16", False, 2),
    ExportConfig("fp16-640-raw", (640, 640), "fp16", False, 4),
    ExportConfig("fp16-640-nms", (640, 640), "fp16", True, 4),
    ExportConfig("fp16-512x768-nms", (512, 768), "fp16", True, 4),
    ExportConfig("int8-512x768-raw", (512, 768), "int8", False, 6),
    ExportConfig("fp16-704x1056-nms", (704, 1056), "fp16", True, 4),
    ExportConfig("fp16-800-nms", (800, 800), "fp16", True, 4),
    ExportConfig("fp16-864-nms", (864, 864), "fp16", True, 4),
    ExportConfig("fp16-896-nms", (896, 896), "fp16", True, 4),
)


def list_presets() -> tuple[ExportConfig, ...]:
    return _PRESETS


def get_preset(name: str) -> ExportConfig:
    for config in _PRESETS:
        if config.preset == name:
            return config
    raise ValueError(
        f"Unknown preset {name!r}; choose from {', '.join(p.preset for p in _PRESETS)}"
    )
