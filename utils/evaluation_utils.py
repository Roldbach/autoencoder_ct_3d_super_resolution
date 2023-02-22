"""Utils for evaluation.

This module contains utility functions for image evaluation.
Most evaluation metrics are utilized from scikit-learn 
image API.
"""
import math

import numpy as np
from skimage import metrics


def evaluate_PSNR(
    input: np.ndarray, label: np.ndarray, data_range: int,
) -> float:
    """Returns Peak Signal-to-Noise Ratio (PSNR).
    
    Returns the Peak Signal-to-Noise Ratio (PSNR) between
    input and label. PSNR focuses on the pixel-level error.
    Data range is also applied for more accurate output.

    Args:
        input:
            A numpy.ndarray that contains pixel values of
            the input image.    
        label:
            A numpy.ndarray that contains pixel values of
            the label image.
        data_range:
            An int that specifies the range of pixel values
            within images. Both images should share the 
            same data range.
    
    Returns:
        A float that specifies the PSNR between input and 
        label.
    """
    return metrics.peak_signal_noise_ratio(input, label, data_range=data_range)

def evaluate_SSIM(
    input: np.ndarray, label: np.ndarray, data_range: int,
) -> float:
    """Returns Structural Similarity Index Measure (SSIM).
    
    Returns the Structural Similarity Index Measure (SSIM)
    between input and label. SSIM focuses on the structural
    difference.  Data range is also applied for more 
    accurate output.

    Args:
        input:
            A numpy.ndarray that contains pixel values of
            the input image.    
        label:
            A numpy.ndarray that contains pixel values of
            the label image.
        data_range:
            An int that specifies the range of pixel values
            within images. Both images should share the 
            same data range.
    
    Returns:
        A float that specifies the SSIM between input and 
        label.
    """
    return metrics.structural_similarity(input, label, data_range=data_range)

def evaluate_RMSE(input: np.ndarray, label: np.ndarray) -> float:
    """Returns Root Mean Square Error (RMSE).
    
    Returns the Root Mean Square Error (RMSE) between input
    and label. RMSE focuses on the pixel-level error.

    Args:
        input:
            A numpy.ndarray that contains pixel values of
            the input image.    
        label:
            A numpy.ndarray that contains pixel values of
            the label image.
    
    Returns:
        A float that specifies the RMSE between input and 
        label.
    """
    return math.sqrt(metrics.mean_squared_error(input, label))