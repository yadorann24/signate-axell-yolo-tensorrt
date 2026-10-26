"""Adapt the pinned exporter to rectangular, read-only INT8 calibration."""

from .calibration import CalibrationData, CalibrationSelection, select_calibration_images
from .config import ExportConfig


def attach_calibration_loader(
    model, calibration: CalibrationData | CalibrationSelection, config: ExportConfig
) -> None:
    """Install a calibration loader on this model's exporter instance only.

    The pinned vendor loader prepares square images even for rectangular engine
    inputs. This adapter retains the requested height and width and reads only
    the verified training images, without label scans, repair, or cache writes.
    """
    config.validate()
    if config.precision != "int8":
        raise ValueError("The calibration adapter requires an INT8 configuration")
    selection = (
        select_calibration_images(calibration, config)
        if isinstance(calibration, CalibrationData)
        else calibration
    )
    if (
        selection.method != config.calibration_selection
        or selection.seed != config.calibration_seed
        or selection.fraction != config.calibration_fraction
    ):
        raise ValueError("calibration selection does not match the export configuration")
    selected_images = selection.images

    def on_export_start(exporter):
        if tuple(exporter.imgsz) != tuple(config.image_size):
            raise ValueError("exporter input shape does not match the calibration configuration")
        if exporter.args.batch != config.batch_size or not exporter.args.int8:
            raise ValueError("exporter must use INT8 with the configured batch size")
        if exporter.args.split != "train" or exporter.args.fraction != config.calibration_fraction:
            raise ValueError(
                "exporter must use the configured training calibration split and fraction"
            )

        def get_calibration_dataloader(prefix=""):
            # Imports stay here so plan/list commands require only the base dependencies.
            import cv2
            import numpy as np
            import torch
            from torch.utils.data import DataLoader
            from ultralytics.data.augment import LetterBox

            letterbox = LetterBox(
                new_shape=tuple(config.image_size), auto=False, scaleup=True, stride=32
            )

            class TrainingImages:
                def __len__(self):
                    return len(selected_images)

                def __getitem__(self, index):
                    path = selected_images[index]
                    # Check around the read while preserving imread's handling of
                    # decodable truncated JPEGs. Sources must stay fixed during export.
                    selection.verify_image(index)
                    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    selection.verify_image(index)
                    if bgr is None:
                        raise ValueError(f"Could not decode calibration training image: {path}")
                    padded = letterbox(image=bgr)
                    rgb_chw = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
                    if rgb_chw.shape != (3, *config.image_size):
                        raise ValueError(f"Unexpected calibration image shape: {path}")
                    # The vendor TensorRT calibrator performs float conversion and /255.
                    return {"img": torch.from_numpy(rgb_chw)}

            return DataLoader(
                TrainingImages(),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=True,
            )

        exporter.get_int8_calibration_dataloader = get_calibration_dataloader

    model.add_callback("on_export_start", on_export_start)
