"""The eight observed YOLO architectures, without competition checkpoints."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    scale: str
    yaml_name: str


_MODELS = tuple(
    ModelSpec(family + scale, family, scale, family + scale + ".yaml")
    for family, scales in (("yolo11", ("n", "s", "m", "l", "x")), ("yolo12", ("n", "m", "l")))
    for scale in scales
)


def list_models() -> tuple[ModelSpec, ...]:
    return _MODELS


def get_model_spec(name: str) -> ModelSpec:
    for model in _MODELS:
        if model.name == name:
            return model
    raise ValueError(f"Unknown model {name!r}; choose from {', '.join(m.name for m in _MODELS)}")


def build_model(name: str, *, num_classes: int = 91):
    """Construct an untrained PyTorch detector from an installed Ultralytics YAML.

    This does not load pretrained weights or perform training. TensorRT export
    instead accepts the caller's existing, trusted .pt checkpoint.
    """
    spec = get_model_spec(name)
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        from ultralytics.nn.tasks import DetectionModel
    except ImportError as error:
        raise RuntimeError(
            "Model construction requires the runtime extra: pip install -e '.[runtime]'"
        ) from error
    return DetectionModel(spec.yaml_name, nc=num_classes, verbose=False)
