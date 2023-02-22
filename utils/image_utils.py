"""Utils for image.

This module contains utility functions for image processing.
Most operations are torch-based for better consistency.
"""
import numpy as np
import torch
from torch import nn


def convert_array_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Converts an image array to an image tensor.

    Converts an image array to an image tensor by expanding
    2 dimensions at front. 

    Args:
        image:
            A numpy.ndarray that contains pixel values of
            the image.
    
    Returns:
        A torch.Tensor that contains pixel values of the
        image.
    """
    for i in range(2):
        image = np.expand_dims(image, axis=0)
    
    return torch.from_numpy(image)

def convert_tensor_to_array(image: torch.Tensor) -> np.ndarray:
    """Converts an iamge tensor to an image array.
    
    Converts an image tensor to an image array after:
        (1) shrinking the first 2 dimensions
        (2) moving the tensor to CPU
        (3) detaching the tensor from the graph
        (4) converting the tensor to an array

    This utility function could be applied to image tensors
    on arbitrary devices.

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the image.
    
    Returns:
        A numpy.ndarray that contains pixel values of the
        image.
    """
    return image[0, 0].cpu().detach().numpy()

def normalise_pixel(
    image: torch.Tensor,
    min: float | None = None,
    max: float | None = None,
    data_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalises image pixel values into [0, 1].

    Normalises pixel values of an image into [0, 1]. If an
    additional range is given, the image is clamped first.
    Otherwise the range is obtained from the given tensor.
    All outputs are casted to float32.

    Args:
        image:
            A torch.Tensor that contains original pixel
            values of the image.
        min:
            A float | None that specifies the minimum
            pixel value.
        max:
            A float | None that specifies the maximum
            pixel value.
        data_type:
            A torch.dtype that specifies the data type
            of pixel values in the output.
        
    Returns:
        A torch.Tensor that contains normalised pixel
        values as float32 of the image.
    """
    min = torch.amin(image) if min is None else min
    max = torch.amax(image) if max is None else max

    image = torch.clamp(image, min, max)
    image = (image-min) / (max-min)

    return image.type(data_type)

def recover_pixel(
    image: torch.Tensor, min: float, max: float, data_type: torch.dtype,
) -> torch.Tensor:
    """Recovers image pixel values back to [min, max].

    Recovers pixel values of an image back to [min, max].
    The output is also casted to the given data type.

    Args:
        image:
            A torch.Tensor that contains normalised pixel
            values of the image.
        min:
            A float that specifies the minimum pixel value
            originally.
        max:
            A float that specifies the maximum pixel value
            originally.
        data_type:
            A torch.dtype that specifies the original data
            type of pixel values in the image.
    
    Returns:
        A torch.Tensor that contains original pixel values
        of the image.
    """
    image = image*(max-min) + min
    
    return image.type(data_type)

def truncate_image(image: torch.Tensor, divisor: int) -> torch.Tensor:
    """Truncates all dimensions within an image.

    Truncates all dimensions within an image so that every
    dimension is divisible by the given divisor.

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the image.
        divisor: 
            An int that specifies the value of the divisor.

    Returns:
        A torch.Tensor that all dimensions are divisble by
        the divisor.
    """
    slice_truncated_all = []
    for index, dimension_size in enumerate(image.shape):
        if index < 2:  # Ignore both batch_number and channel_number
            slice_truncated_all.append(slice(None))
        else:
            slice_truncated_all.append(
                _build_slice_truncated(dimension_size, divisor))

    return image[slice_truncated_all]

def _build_slice_truncated(dimension_size: int, divisor: int) -> slice:
    """Returns a slice after truncation.

    Returns a slice object that indexes the remaining part
    after truncation in the image. Each dimension is
    truncated evenly at the beginning and end.

    Args:
        dimension_size:
            An int that specifies the original dimension of
            the image.
        divisor:
            An int that specifies the value of the divisor.
    
    Returns:
        A slice that indexes the remaining part after
        trunction in the image.
    """
    if dimension_size % divisor == 0:
        return slice(None)
    else:
        truncation = dimension_size % divisor
        truncation_before = truncation // 2
        truncation_after = truncation - truncation_before

        return slice(truncation_before, -truncation_after)

def downsample_image_x_y_axis(
    image: torch.Tensor, scale_factor: int = 2,
) -> torch.Tensor:
    """Downsample both x and y axes of an image.

    Downsample both x and y axes of an image with the given
    scale factor using blinear interpolation.

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the image.
        scale_factor:
            An int that specifies the downsampling factor.
    
    Returns:
        A torch.Tensor that contains pixel values of the
        downsampled image.
    """
    shape = image.shape
    shape_new = (
        shape[0], 
        shape[1], 
        shape[2], 
        shape[3] // scale_factor,
        shape[4] // scale_factor,
    )

    output = torch.zeros(shape_new, dtype=torch.float32)
    for i in range(output.shape[2]):
        output[:, :, i, :, :] = nn.functional.interpolate(
            image[:, :, i, :, :], scale_factor=1/scale_factor, mode='bilinear')
    
    return output

def downsample_image_z_axis(
    image: torch.Tensor, scale_factor: int,
) -> torch.Tensor:
    """Downsample the z axis of an image.

    Downsample the z axis of an image with the given
    scale factor by directly taking away slices at a
    constant interval.

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the image.
        scale_factor:
            An int that specifies the downsampling factor.
    
    Returns:
        A torch.Tensor that contains pixel values of the
        downsampled image.
    """
    return image[:, :, ::scale_factor, :, :]

def upsample_image_z_axis(
    image: torch.Tensor, scale_factor: int, upsample_name: str,
) -> torch.Tensor:
    """Upsample the z axis of an image.

    Upsample the z axis of an image with the given
    scale factor by using the method specified by the name.
    There are 2 methods available:
        (1) 'trilinear_interpolation'
        (2) 'same_insertion'

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the image.
        scale_factor:
            An int that specifies the upsampling factor.
        upsample_name:
            A str that specifies the name of the upsampling
            method.
    
    Returns:
        A torch.Tensor that contains pixel values of the
        upsampled image.
    
    Raises:
        ValueError: The given upsample_name is invalid
    """
    match upsample_name:
        case 'trilinear_interpolation':
            return _upsample_image_z_axis_trilinear_interpolation(
                image, scale_factor)
        case 'same_insertion':
            return _upsample_image_z_axis_same_insertion(
                image, scale_factor)
        case _:
            raise ValueError(f'Can not find upsampling method: {upsample_name}')

def _upsample_image_z_axis_trilinear_interpolation(
    image: torch.Tensor, scale_factor: int,
) -> torch.Tensor:
    """Upsample the z axis of an image using trilinear
    interpolation.

    Upsample the z axis of an image with the given
    scale factor by using trilinear interpolation. The 
    output is casted to float32.

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the image.
        scale_factor:
            An int that specifies the upsampling factor.
    
    Returns:
        A torch.Tensor that contains pixel values as
        float32 of the upsampled image.
    """
    output = nn.functional.interpolate(
        image, scale_factor=(scale_factor, 1, 1), mode='trilinear')

    return output.type(torch.float32)

def _upsample_image_z_axis_same_insertion(
    image: torch.Tensor, scale_factor: int,
) -> torch.Tensor:
    """Upsample the z axis of an image using same insertion.

    Upsample the z axis of an image with the given
    scale factor by the same content at the previous
    position.

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the image.
        scale_factor:
            An int that specifies the upsampling factor.
    
    Returns:
        A torch.Tensor that contains pixel values of the
        upsampled image.
    """
    shape = image.shape
    shape_new = (
        shape[0],
        shape[1],
        shape[2] * scale_factor,
        shape[3],
        shape[4],
    )

    output = torch.zeros(shape_new, dtype=image.dtype)
    for i in range(output.shape[2]):
        output[:, :, i, :, :] = image[:, :, i//scale_factor, :, :]
    
    return output

def patch_image(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patches an image in a non-overlapping manner.

    Converts an image into non-overlapping cubic patches
    with the given size. This utility function could only be
    applied to 3D images.

    Args:
        image:
            A torch.Tensor that contains pixel values of
            the 3D image.
        patch_size:
            An int that specifies the size of cubic
            patches.
        
    Returns:
        A torch.Tensor that contains pixel values of
        all non-overlapping patches.    
    """
    channel_number = image.shape[1]

    image = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    image = image.reshape(
        (-1, channel_number, patch_size, patch_size, patch_size))

    return image