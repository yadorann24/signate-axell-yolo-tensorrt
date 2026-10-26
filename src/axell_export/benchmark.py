"""Measure local detector inference with explicit inputs and synchronized timing."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import time

from .exporting import _new_run_path, _sha256, _timestamp, _write_manifest


@dataclass(frozen=True)
class BenchmarkConfig:
    image_size: tuple[int, int] = (640, 640)
    device: str = "0"
    warmup: int = 10
    repeats: int = 10
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 5
    half: bool = False

    def validate(self) -> None:
        if len(self.image_size) != 2 or any(
            type(n) is not int or n < 32 or n % 32 for n in self.image_size
        ):
            raise ValueError("image_size must be (height, width), positive multiples of 32")
        if self.device not in ("cpu", "0"):
            raise ValueError("device must be cpu or 0")
        for name in ("warmup", "repeats", "max_det"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("conf", "iou"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            minimum_ok = value > 0 if name == "conf" else value >= 0
            if not math.isfinite(value) or not minimum_ok or value > 1:
                raise ValueError("conf must be in (0, 1] and iou in [0, 1]")
        if type(self.half) is not bool or (self.half and self.device == "cpu"):
            raise ValueError("half must be boolean and requires CUDA")

    def predict_kwargs(self) -> dict:
        return {
            "imgsz": list(self.image_size), "rect": False, "batch": 1,
            "device": self.device, "conf": self.conf, "iou": self.iou,
            "max_det": self.max_det, "half": self.half, "augment": False,
            "agnostic_nms": False, "save": False, "save_txt": False,
            "save_crop": False, "verbose": False, "stream": False,
        }


def _image_inventory(list_path: Path) -> list[dict]:
    entries = []
    seen = set()
    for value in list_path.read_text(encoding="utf-8").splitlines():
        value = value.strip()
        if not value:
            continue
        if "://" in value:
            raise ValueError("image list accepts local files only")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = list_path.parent / path
        path = path.resolve()
        if not path.is_file() or path in seen:
            raise ValueError("image list contains a missing file or duplicate image")
        seen.add(path)
        entries.append({"image_id": value, "path": str(path), "sha256": _sha256(path)})
    if not entries:
        raise ValueError("image list must not be empty")
    return entries


def _read_image(entry: dict):
    import cv2
    import numpy as np

    contents = Path(entry["path"]).read_bytes()
    if hashlib.sha256(contents).hexdigest() != entry["sha256"]:
        raise ValueError("benchmark image changed after the inventory was recorded")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {entry['image_id']}")
    return image


def _export_context(weights: Path, config: BenchmarkConfig, manifest_path) -> dict | None:
    if weights.suffix.lower() == ".pt":
        if manifest_path is not None:
            raise ValueError("export manifest is only used with an engine")
        return None
    if config.device != "0":
        raise ValueError("TensorRT engines require CUDA device 0")
    if config.half:
        raise ValueError("engine precision is set at export; omit --half")
    if manifest_path is None:
        raise ValueError("engine benchmarking requires its --export-manifest")
    path = Path(manifest_path).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "complete" or document.get("backend") != "ultralytics":
        raise ValueError("export manifest must record a completed Ultralytics export")
    if document.get("engine", {}).get("sha256") != _sha256(weights):
        raise ValueError("engine hash does not match its export manifest")
    requested = document["requested_config"]
    if tuple(requested["image_size"]) != config.image_size or requested["batch_size"] != 1:
        raise ValueError("benchmark shape and batch must match the exported engine")
    if requested["nms"]:
        # Older exports used the pinned vendor defaults but did not record them.
        conf = requested.get("conf", 0.25)
        iou = requested.get("iou", 0.7)
        max_det = requested.get("max_det", 300)
        if config.conf != conf or config.iou != iou or config.max_det > max_det:
            raise ValueError(
                "requested thresholds cannot replace embedded NMS; use matching conf/iou "
                "and max_det <= the export value, or re-export a raw engine"
            )
    return {
        "path": str(path), "sha256": _sha256(path), "requested_config": requested,
        "source_weights_sha256": document.get("source_weights_sha256"),
    }


def _load_runtime(weights: Path, config: BenchmarkConfig):
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    if metadata.version("ultralytics") != "8.3.170":
        raise RuntimeError("benchmark requires ultralytics==8.3.170")
    import torch
    from ultralytics import YOLO

    if config.device == "0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is not available")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "ultralytics": metadata.version("ultralytics"), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if config.device == "0" else "cpu",
    }
    for package in ("torchvision", "numpy", "opencv-python", "tensorrt-cu12-bindings"):
        try:
            environment[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    synchronize = (lambda: torch.cuda.synchronize(0)) if config.device == "0" else (lambda: None)
    model = YOLO(str(weights), task="detect")
    if model.task != "detect":
        raise ValueError("benchmark requires an object detection checkpoint")
    return model, synchronize, environment


def _detections(result, max_det: int) -> list[dict]:
    import numpy as np

    if result.boxes is None:
        raise ValueError("expected detection boxes from the model")
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confidence = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    if not (np.isfinite(xyxy).all() and np.isfinite(confidence).all()
            and np.isfinite(classes).all()):
        raise ValueError("model returned non-finite predictions")
    if np.any((confidence < 0) | (confidence > 1)) or np.any(classes != classes.astype(int)):
        raise ValueError("model returned invalid confidence or class IDs")
    order = np.argsort(-confidence, kind="stable")[:max_det]
    return [
        {"xyxy": xyxy[i].tolist(), "confidence": float(confidence[i]), "class_id": int(classes[i])}
        for i in order
    ]


def benchmark_model(
    weights: str | Path, image_list: str | Path, output_dir: str | Path,
    config: BenchmarkConfig, *, export_manifest: str | Path | None = None,
    runtime_loader=None,
) -> Path:
    """Save inference timings and restored-coordinate predictions, without AP claims.

    ``runtime_loader`` injects a backend for tests, recorded as such in the report.
    Disk reads, decoding, model loading and serialization are outside timed calls.
    """
    config.validate()
    weights = Path(weights).expanduser().resolve()
    if weights.suffix.lower() not in (".pt", ".engine") or not weights.is_file():
        raise ValueError("weights must be an existing local .pt or .engine file")
    if not weights.stat().st_size:
        raise ValueError("weights must not be empty")
    output_dir = _new_run_path(output_dir)
    image_list = Path(image_list).expanduser().resolve()
    inventory = _image_inventory(image_list)
    export = _export_context(weights, config, export_manifest)
    report = {
        "schema_version": 1, "status": "running", "started_at": _timestamp(),
        "backend": "injected" if runtime_loader is not None else "ultralytics",
        "weights": str(weights), "weights_sha256": _sha256(weights),
        "weights_size_bytes": weights.stat().st_size, "export": export,
        "config": asdict(config), "predict_kwargs": config.predict_kwargs(),
        "images": inventory, "image_list_sha256": _sha256(image_list),
        "timing_scope": {
            "includes": ["predict call overhead", "letterbox/normalization", "host-to-device",
                         "forward", "NMS when needed", "coordinate restoration", "CUDA synchronization"],
            "excludes": ["model loading", "warmup", "image file reads/decoding",
                         "prediction tensor transfer to CPU", "final sort/top-k for saved predictions",
                         "JSON serialization"],
        },
        "metric": "latency only; no accuracy or competition compliance evaluation",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "benchmark.json"
    _write_manifest(path, report)
    try:
        import numpy as np

        loader = _load_runtime if runtime_loader is None else runtime_loader
        model, synchronize, environment = loader(weights, config)
        report["environment"] = environment
        kwargs = config.predict_kwargs()
        warmup_image = _read_image(inventory[0])
        for _ in range(config.warmup):
            model.predict(source=warmup_image.copy(), **kwargs)
        synchronize()
        timings = []
        predictions = []
        for repeat in range(config.repeats):
            for entry in inventory:
                image = _read_image(entry)
                synchronize()
                start = time.perf_counter_ns()
                result = model.predict(source=image, **kwargs)
                synchronize()
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                if len(result) != 1:
                    raise ValueError("expected exactly one result for batch=1")
                timings.append({"image_id": entry["image_id"], "repeat": repeat,
                                "elapsed_ms": elapsed_ms})
                if repeat == 0:
                    predictions.append({
                        "image_id": entry["image_id"], "height": image.shape[0],
                        "width": image.shape[1],
                        "detections": _detections(result[0], config.max_det),
                    })
        if _sha256(weights) != report["weights_sha256"]:
            raise ValueError("weights changed during benchmark")
        values = [row["elapsed_ms"] for row in timings]
        report["latency_ms"] = {
            "count": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "min": min(values), "max": max(values),
        }
        report["class_names"] = model.names
        report["timings"] = timings
        report["predictions"] = predictions
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_at"] = _timestamp()
        _write_manifest(path, report)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path, help="local image list, one path per line")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 640), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--device", choices=("cpu", "0"), default="0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args(argv)
    config = BenchmarkConfig(tuple(args.image_size), args.device, args.warmup, args.repeats,
                             args.conf, args.iou, args.max_det, args.half)
    try:
        path = benchmark_model(args.weights, args.images, args.output_dir, config,
                               export_manifest=args.export_manifest)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
