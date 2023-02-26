"""Utils for evaluation.

This module contains utility functions for image evaluation.
Most evaluation metrics are utilized from scikit-learn 
image API.
"""
import math

import numpy as np
from skimage import metrics


def evaluate_PSNR(
    candidate: np.ndarray, reference: np.ndarray, data_range: int) -> float:
    """Returns Peak Signal-to-Noise Ratio (PSNR).
    
    Returns the Peak Signal-to-Noise Ratio (PSNR) between
    candidate and reference. PSNR focuses on the pixel-level
    error.  Data range is also applied for more accurate
    output.

    Args:
        candidate:
            A numpy.ndarray that contains pixel values of
            the candidate.    
        reference:
            A numpy.ndarray that contains pixel values of
            the reference.
        data_range:
            An int that specifies the range of pixel values
            within images. Both images should share the 
            same data range.
    
    Returns:
        A float that specifies the PSNR between the
        candidate and the reference.
    """
    return metrics.peak_signal_noise_ratio(
        candidate, reference, data_range=data_range)

def evaluate_SSIM(
    candidate: np.ndarray, reference: np.ndarray, data_range: int) -> float:
    """Returns Structural Similarity Index Measure (SSIM).
    
    Returns the Structural Similarity Index Measure (SSIM)
    between candidate and reference. SSIM focuses on the
    structural difference.  Data range is also applied for
    more accurate output.

    Args:
        candidate:
            A numpy.ndarray that contains pixel values of
            the candidate.    
        reference:
            A numpy.ndarray that contains pixel values of
            the reference.
        data_range:
            An int that specifies the range of pixel values
            within images. Both images should share the 
            same data range.
    
    Returns:
        A float that specifies the SSIM between the
        candidate and the reference.
    """
    return metrics.structural_similarity(
        candidate, reference, data_range=data_range)

def evaluate_RMSE(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Returns Root Mean Square Error (RMSE).
    
    Returns the Root Mean Square Error (RMSE) between candidate
    and reference. RMSE focuses on the pixel-level error.

    Args:
        candidate:
            A numpy.ndarray that contains pixel values of
            the candidate.    
        reference:
            A numpy.ndarray that contains pixel values of
            the reference.
    
    Returns:
        A float that specifies the RMSE between candidate and 
        reference.
    """
    return math.sqrt(metrics.mean_squared_error(candidate, reference))