"""Helper of model training/validation.

This module contains classes and functions that can
facilitate the model training/validation.
"""
import os
import shutil

import pandas as pd
import torch
from torch import nn

from configuration import device_configuration
from utils import image_utils


class BatchExtractor:
    """An iterator for efficient batch extraction.

    An iterator that can efficiently extract batches from
    the given input/label pair. This is specifically used
    when images are patched and can not be directly used in
    batch operations.
    """
    def __init__(
        self,
        input: torch.Tensor,
        label: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Initialises BatchExtractor."""
        self._input = input
        self._label = label
        self._batch_size = batch_size
        self._index = 0
        self._limit = self._input.shape[0]
    
    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._index >= self._limit:
            raise StopIteration
        
        next_batch_slice = self._build_next_batch_slice() 

        self._index += self._batch_size

        return self._input[next_batch_slice], self._label[next_batch_slice]
    
    def _build_next_batch_slice(self) -> slice:
        """Returns a slice object to extract the next batch.

        Returns a slice obejct that can be used to correctly
        extract batches by __next__(). The end part is
        treated as an individual mini-batch.

        Returns:
            A slice object that can be used to extract next
            batch.
        """
        if self._index + self._batch_size > self._limit:
            return slice(self._index, None)
        else:
            return slice(self._index, self._index+self._batch_size)

class BatchLossAccumulator:
    """A tool to accumulate the loss for all batches.

    A tool to accumulate the loss for all batches and
    compute the average value as the final epoch loss.
    """
    def __init__(self) -> None:
        """Initialises BatchLossAccumulator."""
        self._count = 0
        self._batch_loss_total = 0
    
    def accumulate_batch_loss(self, batch_loss: float) -> None:
        """Accumulates batch loss."""
        self._count += 1
        self._batch_loss_total += batch_loss
    
    def average_batch_loss(self) -> float:
        """Returns the average batch loss."""
        return self._batch_loss_total / self._count

def reset_directory(directory_path: str) -> None:
    """Resets the directory at the given path.

    Tries to delete the directory at the given path if it
    can be found and creates an empty directory.
    """
    try:
        shutil.rmtree(directory_path)
    except:
        pass
    finally:
        os.mkdir(directory_path)

def validate_epoch(
    input: torch.Tensor,
    label: torch.Tensor,
    window: tuple[float|None, float|None],
    scale_factor: int,
    upsample_name: str,
    patch_size: int,
    batch_size: int,
    model: nn.Module,
    loss_function: nn.Module,
    validation_batch_loss_accumulator: BatchLossAccumulator,
) -> None:
    """Validates the model within an epoch.

    Validates the model within an epoch with the following
    steps:
        (1) Pre-processes input and label.
        (2) Validates the model batch-by-batch.
    
    Args:
        input:
            A torch.Tensor that contains pixel values of
            the input.    
        label:
            A torch.Tensor that contains pixel values of
            the label.
        window:
            A tuple[float|None, float|None] that specifies
            the range of pixel values interested.
        scale_factor:
            An int that specifies the scale factor of
            downsampling/upsampling operations in z axis.
        upsample_name:
            A str that specifies the name of the upsampling
            method.     
        patch_size:
            An int that specifies the size of cubic
            patches.
        batch_size:
            An int that specifies the number of data within
            one batch.
        model:
            A torch.nn.Module that specifies the
            architecture of the model.
        loss_function:
            A torch.nn.Module that defines the difference
            between the prediction and label.
        validation_batch_loss_accumulator:
            A BatchAccumulator that accumulates validation
            loss for all batches.
    """
    input = pre_process_input(
        input, window, scale_factor, upsample_name, patch_size)
    label = pre_process_label(label, window, patch_size)

    for input_batch, label_batch in BatchExtractor(input, label, batch_size):
        validate_batch(
            input_batch, label_batch,
            model,
            loss_function,
            validation_batch_loss_accumulator,
        )

def validate_batch(
    input_batch: torch.Tensor,
    label_batch: torch.Tensor,
    model: nn.Module,
    loss_function: nn.Module,
    validation_batch_loss_accumulator: BatchLossAccumulator,
) -> None:
    """Validates the model using batch data.

    Validates the model using batch data with the following
    steps:
        (1) Moves batch data to the device used for model
            training.
        (2) Builds predition using the given input and
            computes the loss between prediction and label.
        (3) Accumulates the batch loss.
    
    Args:
        input_batch:
            A torch.Tensor that contains pixel values of
            the input batch.    
        label_batch:
            A torch.Tensor that contains pixel values of
            the label batch.
        model:
            A torch.nn.Module that specifies the
            architecture of the model.
        loss_function:
            A torch.nn.Module that defines the difference
            between the prediction and label.
        validation_batch_loss_accumulator:
            A BatchAccumulator that accumulates validation
            loss for all batches.
    """
    input_batch = input_batch.to(device_configuration.TRAIN_DEVICE)
    label_batch = label_batch.to(device_configuration.TRAIN_DEVICE)

    validation_batch_loss = loss_function(model(input_batch), label_batch)

    validation_batch_loss_accumulator.accumulate_batch_loss(
        validation_batch_loss.item())

def train_epoch(
    input: torch.Tensor,
    label: torch.Tensor,
    window: tuple[float|None, float|None],
    scale_factor: int,
    upsample_name: str,
    patch_size: int,
    batch_size: int,
    model: nn.Module,
    optimiser: nn.Module,
    loss_function: nn.Module,
    train_batch_loss_accumulator: BatchLossAccumulator,
) -> None:
    """Validates the model within an epoch.

    Validates the model within an epoch with the following
    steps:
        (1) Pre-processes input and label.
        (2) Validates the model batch-by-batch.
    
    Args:
        input:
            A torch.Tensor that contains pixel values of
            the input.    
        label:
            A torch.Tensor that contains pixel values of
            the label.
        window:
            A tuple[float|None, float|None] that specifies
            the range of pixel values interested.
        scale_factor:
            An int that specifies the scale factor of
            downsampling/upsampling operations in z axis.
        upsample_name:
            A str that specifies the name of the upsampling
            method.     
        patch_size:
            An int that specifies the size of cubic
            patches.
        batch_size:
            An int that specifies the number of data within
            one batch.
        model:
            A torch.nn.Module that specifies the
            architecture of the model.
        optimiser:
            A torch.nn.Module that can be used to update
            model weights.
        loss_function:
            A torch.nn.Module that defines the difference
            between the prediction and label.
        train_batch_loss_accumulator:
            A BatchAccumulator that accumulates train loss
            for all batches.
    """
    input = pre_process_input(
        input, window, scale_factor, upsample_name, patch_size)
    label = pre_process_label(label, window, patch_size)

    for input_batch, label_batch in BatchExtractor(input, label, batch_size):
        train_batch(
            input_batch, label_batch,
            model,
            optimiser,
            loss_function,
            train_batch_loss_accumulator,
        )

def train_batch(
    input_batch: torch.Tensor,
    label_batch: torch.Tensor,
    model: nn.Module,
    optimiser: nn.Module,
    loss_function: nn.Module,
    train_batch_loss_accumulator: BatchLossAccumulator,
) -> None:
    """Trains the model using batch data.

    Trains the model using batch data with the following
    steps:
        (1) Moves batch data to the device used for model
            training.
        (2) Builds predition using the given input and
            computes the loss between prediction and label.
        (3) Accumulates the batch loss.
        (4) Updates model weights based on the batch loss.
    
    Args:
        input_batch:
            A torch.Tensor that contains pixel values of
            the input batch.    
        label_batch:
            A torch.Tensor that contains pixel values of
            the label batch.
        model:
            A torch.nn.Module that specifies the
            architecture of the model.
        optimiser:
            A torch.nn.Module that can be used to update
            model weights.
        loss_function:
            A torch.nn.Module that defines the difference
            between the prediction and label.
        train_batch_loss_accumulator:
            A BatchAccumulator that accumulates train loss
            for all batches.
    """
    input_batch = input_batch.to(device_configuration.TRAIN_DEVICE)
    label_batch = label_batch.to(device_configuration.TRAIN_DEVICE)

    optimiser.zero_grad()
    batch_loss = loss_function(model(input_batch), label_batch)
    train_batch_loss_accumulator.accumulate_batch_loss(batch_loss.item())
    batch_loss.backward()

    optimiser.step()

def pre_process_input(
    input: torch.Tensor,
    window: tuple[float|None, float|None],
    scale_factor: int,
    upsample_name: str,
    patch_size: int,
) -> torch.Tensor:
    """Pre-processes input.

    Pre-processes input before model training/validation
    with the following steps:
        (1) Normalises the input with the given window.
        (2) Truncates the input so its shape can fit
            non-overlapping patching.
        (3) Downsamples the input in x, y axes by x2.
        (4) Downsamples the input in z axis by the given
            scale factor.
        (5) Upsamples the input in z axis using the given
            upsampling method to the original dimension.
        (6) Patches the input in a non-overlapping manner.
    
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
        patch_size:
            An int that specifies the size of cubic
            patches.

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed input.
    """
    input = image_utils.normalise_pixel(input, *window)
    input = image_utils.truncate_image(input, patch_size)
    input = image_utils.downsample_image_x_y_axis(input)
    input = image_utils.downsample_image_z_axis(input, scale_factor)
    input = image_utils.upsample_image_z_axis(
        input, scale_factor, upsample_name)
    input = image_utils.patch_image(input, patch_size)

    return input

def pre_process_label(
    label: torch.Tensor,
    window: tuple[float|None, float|None],
    patch_size: int,
) -> torch.Tensor:
    """Pre-processes label.

    Pre-processes label before model training/validation
    with the following steps:
        (1) Normalises the input with the given window.
        (2) Truncates the input so its shape can fit
            non-overlapping patching.
        (3) Downsamples the input in x, y axes by x2.
        (4) Patches the input in a non-overlapping manner.
    
    Args:
        input:
            A torch.Tensor that contains pixel values of
            the input.
        window:
            A tuple[float|None, float|None] that specifies
            the range of pixel values interested.
        patch_size:
            An int that specifies the size of cubic
            patches.

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed label.
    """
    label = image_utils.normalise_pixel(label, *window)
    label = image_utils.truncate_image(label, patch_size)
    label = image_utils.downsample_image_x_y_axis(label)
    label = image_utils.patch_image(label, patch_size)

    return label

def append_record(
    epoch_data: dict[str, float], record: pd.DataFrame) -> pd.DataFrame:
    """Appends data from current epoch to record."""
    row = {key: [value] for key, value in epoch_data.items()}
    row = pd.DataFrame(row)

    return pd.concat((record, row))

def report_epoch_data(epoch_data: dict[str, float]) -> None:
    """Reports data from current epoch in terminal."""
    print(
        'Epoch:', epoch_data['epoch'], '\t',
        'Train Loss:', '{:.4e}'.format(epoch_data['train_loss']), '\t',
        'Validation Loss:', '{:.4e}'.format(epoch_data['validation_loss']), '\t',
        'Time:', '{:.2f}'.format(epoch_data['time']), 's',
    )