"""Helper of model testing.

This module contains classes and functions that can
facilitate the model testing.
"""
import numpy as np
import torch

from utils import evaluation_utils
from utils import image_utils


def pre_process_input(
    input: torch.Tensor,
    window: tuple[float|None, float|None],
    scale_factor: int,
    upsample_name: str,
) -> torch.Tensor:
    """Pre-processes input.

    Pre-processes input before model testing with the following steps:
        (1) Normalises the input with the given window.
        (2) Truncates the input so its shape can fit
            non-overlapping patching.
        (3) Downsamples the input in x, y axes by x2.
        (4) Downsamples the input in z axis by the given
            scale factor.
        (5) Upsamples the input in z axis using the given
            upsampling method to the original dimension.
    
    Args:
        input:
            A torch.Tensor that contains pixel values of
            the input.
        window:
            A tuple[float|None, float|None] that specifies
            the range of pixel values interested.
        scale_factor:
            An int that specifies the scale factor of
            downsampling/upsampling operations in z axis.
        upsample_name:
            A str that specifies the name of the upsampling
            method.     

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed input.
    """
    input = image_utils.normalise_pixel(input, *window)
    input = image_utils.truncate_image(input, scale_factor)
    input = image_utils.downsample_image_x_y_axis(input)
    input = image_utils.downsample_image_z_axis(input, scale_factor)
    input = image_utils.upsample_image_z_axis(
        input, scale_factor, upsample_name)

    return input

def pre_process_label(
    label: torch.Tensor,
    window: tuple[float|None, float|None],
    scale_factor: int,
) -> torch.Tensor:
    """Pre-processes label.

    Pre-processes label before model testing with the following steps:
        (1) Normalises the input with the given window.
        (2) Truncates the input so its shape can fit
            non-overlapping patching.
        (3) Downsamples the input in x, y axes by x2.
    
    Args:
        input:
            A torch.Tensor that contains pixel values of
            the input.
        window:
            A tuple[float|None, float|None] that specifies
            the range of pixel values interested.
        scale_factor:
            An int that specifies the scale factor of
            downsampling/upsampling operations in z axis

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed label.
    """
    label = image_utils.normalise_pixel(label, *window)
    label = image_utils.truncate_image(label, scale_factor)
    label = image_utils.downsample_image_x_y_axis(label)

    return label

def post_process_image(image: torch.Tensor) -> np.ndarray:
    """Post-processes image.
    
    Post-processes image with the following steps:
        (1) Recovers pixel values to [0, 255] and converts the data type to
            uint8.
        (2) Converts the image from tensor to array.
    """
    image = image_utils.recover_pixel(image, 0, 255, torch.uint8)
    image = image_utils.convert_tensor_to_array(image)

    return image

def evaluate(
    input: np.ndarray,
    label: np.ndarray,
    data_range: int = 255,
) -> tuple[float, float, float]:
    """Evaluates PSNR, SSIM and RMSE between input and label."""
    return (
        evaluation_utils.evaluate_PSNR(input, label, data_range),
        evaluation_utils.evaluate_SSIM(input, label, data_range),
        evaluation_utils.evaluate_RMSE(input, label),
    )

def report_evaluation_result(
    input_vs_label: list[tuple[float, ...]],
    prediction_vs_label: list[tuple[float, ...]],
) -> None:
    """Reports evaluation results in terminal."""
    input_vs_label_mean_std = _build_evaluation_result_mean_std(input_vs_label)
    prediction_vs_label_mean_std = _build_evaluation_result_mean_std(
        prediction_vs_label)

    for index, evaluation_metric in enumerate(('PSNR', 'SSIM', 'RMSE')):
        _report_evaluation_result_mean_std(
            tuple(element[index] for element in input_vs_label_mean_std),
            tuple(element[index] for element in prediction_vs_label_mean_std),
            evaluation_metric,
        )

def _build_evaluation_result_mean_std(
    evaluation_result: list[tuple[float, ...]]
) -> tuple[list[float], list[float]]:
    return (
        _build_evaluation_result_mean(evaluation_result),
        _build_evaluation_result_std(evaluation_result),
    )

def _build_evaluation_result_mean(
    evaluation_result: list[tuple[float, ...]]) -> list[float]:
    return np.mean(evaluation_result, axis=0)

def _build_evaluation_result_std(
    evaluation_result: list[tuple[float, ...]]) -> list[float]:
    return np.std(evaluation_result, axis=0)

def _report_evaluation_result_mean_std(
    input_vs_label_mean_std_single: tuple[float, float],
    prediction_vs_label_mean_std_single: tuple[float, float],
    evaluation_metric: str,
) -> None:
    print(
        '{} Mean: {:.4f} / {:.4f}'.format(
            evaluation_metric,
            prediction_vs_label_mean_std_single[0],
            input_vs_label_mean_std_single[0],
        )
    )
    print(
        '{} STD: {:.4f} / {:.4f}'.format(
            evaluation_metric,
            prediction_vs_label_mean_std_single[1],
            input_vs_label_mean_std_single[1],
        )
    )
    print('')
