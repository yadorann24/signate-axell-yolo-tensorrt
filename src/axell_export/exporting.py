"""Plan and run isolated exports of a local YOLO checkpoint to TensorRT."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil

from .calibration import (
    inspect_calibration_data,
    select_calibration_images,
    write_calibration_data,
)
from .config import ExportConfig


EXPECTED_RUNTIME = {
    "ultralytics": "8.3.170",
    "tensorrt-cu12-bindings": "10.13.0.35",
    "onnx": "1.16.2",
    "onnxslim": "0.1.59",
}


def _source_weights(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError("weights must be an existing local .pt checkpoint")
    if path.stat().st_size == 0:
        raise ValueError("weights checkpoint must not be empty")
    return path


def _new_run_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"output directory must be new: {raw}")
    return raw.resolve()


def plan_export(
    weights: str | Path,
    output_dir: str | Path,
    config: ExportConfig,
    *,
    calibration_data: str | Path | None = None,
    device: int = 0,
) -> dict:
    """Validate inputs without importing a model runtime or creating output files."""
    config.validate()
    if isinstance(device, bool) or not isinstance(device, int) or device != 0:
        raise ValueError("this export workflow supports CUDA device 0 only")
    checkpoint = _source_weights(weights)
    run_dir = _new_run_path(output_dir)
    calibration = None
    if config.precision == "int8":
        if calibration_data is None:
            raise ValueError("INT8 export requires a local calibration YAML with a train split")
        calibration = select_calibration_images(
            inspect_calibration_data(calibration_data), config
        ).summary()
    elif calibration_data is not None:
        raise ValueError("calibration data is only accepted for INT8 export")
    kwargs = {**config.export_kwargs(), "device": device}
    if calibration is not None:
        kwargs.update(data=str(run_dir / "calibration.yaml"), split="train")
    return {
        "weights": str(checkpoint),
        "output_dir": str(run_dir),
        "device": device,
        "requested_config": config.to_dict(),
        "export_kwargs": kwargs,
        "calibration": calibration,
    }


def _runtime_loader(device: int):
    """Check the recorded GPU export stack before loading any checkpoint."""
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    versions = {}
    for package, expected in EXPECTED_RUNTIME.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"missing export dependency: {package}=={expected}") from error
        if actual != expected:
            raise RuntimeError(f"{package}=={expected} is required; found {actual}")
        versions[package] = actual
    for package in ("torch", "torchvision", "onnxruntime-gpu"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"missing export dependency: {package}") from error
    import torch

    # Keep TensorRT import after ONNX conversion, as required by the vendor exporter.
    # Its binding distribution version has already been checked above.
    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is not available for TensorRT export")
    versions["cuda"] = torch.version.cuda
    versions["gpu"] = torch.cuda.get_device_name(device)
    from ultralytics import YOLO

    return YOLO, versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, document: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_engine(
    weights: str | Path,
    output_dir: str | Path,
    config: ExportConfig,
    *,
    calibration_data: str | Path | None = None,
    device: int = 0,
    model_loader=None,
) -> Path:
    """Export in a new run directory and return its ``export.json`` manifest.

    ``model_loader`` is an injectable backend for tests. Supplying it bypasses
    CUDA/dependency checks; the manifest identifies such a run explicitly.
    Checkpoints are copied into the run to isolate vendor-generated artifacts.
    """
    plan = plan_export(
        weights, output_dir, config, calibration_data=calibration_data, device=device
    )
    if model_loader is None:
        model_loader, environment = _runtime_loader(device)
        backend = "ultralytics"
    else:
        environment = {}
        backend = "injected"
    checkpoint = Path(plan["weights"])
    run_dir = Path(plan["output_dir"])
    calibration = (
        select_calibration_images(inspect_calibration_data(calibration_data), config)
        if calibration_data is not None
        else None
    )
    if calibration is not None:
        for field, value in calibration.summary().items():
            if plan["calibration"][field] != value:
                raise ValueError(
                    "calibration inputs changed during preflight; plan the export again"
                )
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "export.json"
    work_checkpoint = run_dir / "model.pt"
    document = {
        **plan,
        "schema_version": 2,
        "status": "running",
        "started_at": _timestamp(),
        "backend": backend,
        "environment": environment,
    }
    try:
        _write_manifest(manifest_path, document)
        shutil.copy2(checkpoint, work_checkpoint)
        document["source_weights_sha256"] = _sha256(work_checkpoint)
        kwargs = {**config.export_kwargs(), "device": device}
        if calibration is not None:
            kwargs["data"] = str(write_calibration_data(calibration, run_dir))
            kwargs["split"] = "train"
        document["export_kwargs"] = kwargs
        _write_manifest(manifest_path, document)
        model = model_loader(str(work_checkpoint))
        if calibration is not None and backend == "ultralytics":
            from .runtime import attach_calibration_loader

            attach_calibration_loader(model, calibration, config)
            document["calibration"]["loader"] = "readonly_rgb_letterbox"
            document["calibration"]["tensor_shape"] = [1, 3, *config.image_size]
            _write_manifest(manifest_path, document)
        result = model.export(**kwargs)
        if calibration is not None:
            calibration.verify_contents()
        if not isinstance(result, (str, Path)):
            raise RuntimeError("export backend did not return a single engine file path")
        engine = Path(result).expanduser()
        expected_engine = run_dir / "model.engine"
        if engine.resolve() != expected_engine or engine.is_symlink():
            raise RuntimeError("export backend returned an unexpected engine path")
        if not engine.is_file() or engine.stat().st_size == 0:
            raise RuntimeError("export backend did not create a nonempty engine file")
        document["engine"] = {
            "path": str(expected_engine),
            "sha256": _sha256(engine),
            "size_bytes": engine.stat().st_size,
        }
        work_checkpoint.unlink()
        document["status"] = "complete"
        document["finished_at"] = _timestamp()
        _write_manifest(manifest_path, document)
    except BaseException as error:
        try:
            work_checkpoint.unlink(missing_ok=True)
        except OSError as cleanup_error:
            document["cleanup_error"] = str(cleanup_error)
        document["status"] = "failed"
        document["finished_at"] = _timestamp()
        document["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            _write_manifest(manifest_path, document)
        except OSError as recording_error:
            error.add_note(f"Could not record the failed export: {recording_error}")
        raise
    return manifest_path
