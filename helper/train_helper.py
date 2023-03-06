"""Helper of model training/validation.

This module contains classes and functions that can
facilitate the model training/validation. The main function
train() is compatible with arguments from the terminal.
"""
from argparse import Namespace
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from configuration import device_configuration
from dataset.AapmMayoDataset import AapmMayoDataset
from model.Autoencoder import Autoencoder_Conv
from model.Autoencoder import Autoencoder_Maxpool
from model.PlainCNN import PlainCNN
from model.UNet import UNet
from utils import image_utils
from utils import io_utils
from utils import path_utils


class TrainDelegate:
    """A delegate designed for model training only.

    A delegate that is specifically designed for model
    training. It contains all components required and 
    some convenient functions that fascilitate training
    experiments. Arguments are directly obtained from the
    terminal.
    """

    def __init__(self, argument: Namespace) -> None:
        """Initialises TrainDelegate."""
        self._weight_file_path = self._construct_weight_file_path(argument)
        self._record_file_path = self._construct_record_file_path(argument)
        self._train_data_loader = self._construct_train_data_loader()
        self._validation_data_loader = self._construct_validation_data_loader()
        self._model = self._construct_model(argument)
        self._optimiser = self._construct_optimiser(argument)    
        self._loss_function = self._construct_loss_function()
        self._record = self._construct_record()

        if argument.is_resuming:
            self._read_weight_and_record()
        else:
            self._reset_weight_and_record(argument)

    def _construct_weight_file_path(self, argument: Namespace) -> str:
        """Constructs a file path to store model weights."""
        return '{}/{}_{}_x{}.pth'.format(
            argument.output_directory_path,
            argument.model_name,
            argument.upsample_name,
            argument.scale_factor,
        )

    def _construct_record_file_path(self, argument: Namespace) -> str:
        """Constructs a file path to store record."""
        return f'{argument.output_directory_path}/record.csv'

    def _construct_train_data_loader(self) -> DataLoader:
        """Constructs a dataloader to load train data."""
        return DataLoader(
            dataset = AapmMayoDataset('train'),
            batch_size = 1,
            shuffle = True,
            num_workers = 4,
        )

    def _construct_validation_data_loader(self) -> DataLoader:
        """Constructs a dataloader to load validation data."""
        return DataLoader(
            dataset = AapmMayoDataset('validation'),
            batch_size = 1,
            shuffle = True,
            num_workers = 4,
        )
    
    def _construct_model(self, argument: Namespace) -> nn.Module:
        """Constructs a model used in the experiment.

        Constructs a model based on the given model name.
        The model is moved to the train device and also
        coverted to the data parallelism if multiple GPUs are
        available.

        Args:
            argument:
                A argparse.Namespace that contains arguments
                directly from the terminal.

        Returns:
            A torch.nn.Module that specifies the corresponding
            architecture of the model.
        
        Raises:
            ValueError: The given model_name is invalid.    
        """
        device = device_configuration.TRAIN_DEVICE
        device_id = device_configuration.TRAIN_DEVICE_ID

        match argument.model_name:
            case 'PlainCNN':
                model = PlainCNN().to(device)
            case 'AE_Maxpool':
                model = Autoencoder_Maxpool().to(device)
            case 'AE_Conv':
                model = Autoencoder_Conv().to(device)
            case 'UNet':
                model = UNet().to(device)
            case _:
                raise ValueError(
                    f'Can not find model: {argument.model_name}')

        if device_id is not None:
            model = nn.parallel.DataParallel(model, device_id)
        
        return model
    
    def _construct_optimiser(self, argument: Namespace) -> nn.Module:
        """Constructs an Adam optimiser."""
        return torch.optim.Adam(self._model.parameters(), argument.learning_rate)
        
    def _construct_loss_function(self) -> nn.Module:
        """Constructs a L2 loss function."""
        return nn.MSELoss()

    def _construct_record(self) -> pd.DataFrame:
        """Constructs an empty record."""
        return pd.DataFrame()
    
    def _read_weight_and_record(self) -> None:
        """Reads weights and record from last experiment."""
        self._model = io_utils.read_weight(self._weight_file_path, self._model)
        self._record = io_utils.read_file(self._record_file_path)
    
    def _write_weight_and_record(self) -> None:
        """Writes current weights and record."""
        io_utils.write_weight(self._weight_file_path, self._model)
        io_utils.write_file(self._record_file_path, self._record)
    
    def _reset_weight_and_record(self, argument: Namespace) -> None:
        """Resets outputs from last experiment."""
        path_utils.make_directory(argument.output_directory_path)
        path_utils.reset_file(self._weight_file_path)
        path_utils.reset_file(self._record_file_path)

    def _is_finished(self, epoch: int) -> bool:
        """Returns whether the training is finished or not."""
        return len(self._record) >= epoch

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

def train(argument: Namespace) -> None:
    """Trains the model using given experiment settings.

    Trains the model using given experiment settings by the
    following steps:
        (1) Validates the model within an epoch.
        (2) Trains the model within an epoch.
        (3) Append summary of the current epoch to record.
        (3) Writes current weights and records.
        (4) Report epoch summary to the terminal.
        (5) Repeats step 1~4 until finishing.
    
    Args:
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal.
    """
    delegate = TrainDelegate(argument)

    while not delegate._is_finished(argument.epoch):
        time_start = time.time()
        validation_loss = validate_epoch(delegate, argument)
        train_loss = train_epoch(delegate, argument)
        time_end = time.time()

        epoch_summary = {
            'epoch': len(delegate._record) + 1,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'time': time_end - time_start,
        }
        append_record(epoch_summary, delegate)
        report_epoch_summary(epoch_summary)

        delegate._write_weight_and_record()

        torch.cuda.empty_cache()

def validate_epoch(delegate: TrainDelegate, argument: Namespace) -> float:
    """Validates the model within an epoch.

    Validates the model within an epoch with the following
    steps:
        (1) Pre-processes input and label.
        (2) Validates the model batch-by-batch.
    
    Args:
        delegate:
            A TrainDelegate that contains all components
            required in model training.
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal. 

    Returns:
        A float that represents the validation loss across
        all batches at the current epoch.
    """
    batch_loss_accumulator = BatchLossAccumulator()

    with torch.no_grad():
        delegate._model.eval()

        for input, label in delegate._validation_data_loader:
            input = pre_process_input(input, argument)
            label = pre_process_label(label, argument)

            batch_extractor = BatchExtractor(input, label, argument.batch_size)
            for input_batch, label_batch in batch_extractor:
                validate_batch(
                    input_batch, label_batch, delegate, batch_loss_accumulator)
    
    return batch_loss_accumulator.average_batch_loss()

def validate_batch(
    input_batch: torch.Tensor,
    label_batch: torch.Tensor,
    delegate: TrainDelegate,
    batch_loss_accumulator: BatchLossAccumulator,
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
        delegate:
            A TrainDelegate that contains all components
            required in model training.
        batch_loss_accumulator:
            A BatchLossAccumulator that can accumulate the
            loss for all batches and compute the average
            value as the final epoch loss.
    """
    input_batch = input_batch.to(device_configuration.TRAIN_DEVICE)
    label_batch = label_batch.to(device_configuration.TRAIN_DEVICE)

    prediction_batch = delegate._model(input_batch)

    batch_loss = delegate._loss_function(prediction_batch, label_batch)
    batch_loss_accumulator.accumulate_batch_loss(batch_loss.item())

def train_epoch(delegate: TrainDelegate, argument: Namespace) -> float:
    """Trains the model within an epoch.

    Trains the model within an epoch with the following
    steps:
        (1) Pre-processes input and label.
        (2) Trains the model batch-by-batch.
    
    Args:
        delegate:
            A TrainDelegate that contains all components
            required in model training.
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal.

    Returns:
        A float that represents the train loss across all
        batches at the current epoch.
    """
    batch_loss_accumulator = BatchLossAccumulator()

    delegate._model.train()
    for input, label in delegate._train_data_loader:
        input = pre_process_input(input, argument)
        label = pre_process_label(label, argument)

        batch_extractor = BatchExtractor(input, label, argument.batch_size)
        for input_batch, label_batch in batch_extractor:
            train_batch(
                input_batch, label_batch, delegate, batch_loss_accumulator)
    
    return batch_loss_accumulator.average_batch_loss()

def train_batch(
    input_batch: torch.Tensor,
    label_batch: torch.Tensor,
    delegate: TrainDelegate,
    batch_loss_accumulator: BatchLossAccumulator,
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
        delegate:
            A TrainDelegate that contains all components
            required in model training.
        batch_loss_accumulator:
            A BatchLossAccumulator that can accumulate the
            loss for all batches and compute the average
            value as the final epoch loss.
    """
    input_batch = input_batch.to(device_configuration.TRAIN_DEVICE)
    label_batch = label_batch.to(device_configuration.TRAIN_DEVICE)

    delegate._optimiser.zero_grad()
    prediction_batch = delegate._model(input_batch)
    batch_loss = delegate._loss_function(prediction_batch, label_batch)
    batch_loss.backward()
    delegate._optimiser.step()

    batch_loss_accumulator.accumulate_batch_loss(batch_loss.item())

def pre_process_input(
    input: torch.Tensor, argument: Namespace) -> torch.Tensor:
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
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal.

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed input.
    """
    input = image_utils.normalise_pixel(input, *argument.window)
    input = image_utils.truncate_image(input, argument.patch_size)
    input = image_utils.downsample_image_x_y_axis(input)
    input = image_utils.downsample_image_z_axis(input, argument.scale_factor)
    input = image_utils.upsample_image_z_axis(
        input, argument.scale_factor, argument.upsample_name)
    input = image_utils.patch_image(input, argument.patch_size)

    return input

def pre_process_label(
    label: torch.Tensor, argument: Namespace) -> torch.Tensor:
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
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal.

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed label.
    """
    label = image_utils.normalise_pixel(label, *argument.window)
    label = image_utils.truncate_image(label, argument.patch_size)
    label = image_utils.downsample_image_x_y_axis(label)
    label = image_utils.patch_image(label, argument.patch_size)

    return label

def append_record(
    epoch_summary: dict[str, float], delegate: TrainDelegate) -> None:
    """Appends data from current epoch to record."""
    row = {key: [value] for key, value in epoch_summary.items()}
    row = pd.DataFrame(row)

    delegate._record = pd.concat((delegate._record, row))

def report_epoch_summary(epoch_summary: dict[str, float]) -> None:
    """Reports summary of the current epoch in the terminal."""
    print(
        'Epoch:', epoch_summary['epoch'], '\t',
        'Train Loss:', '{:.4e}'.format(epoch_summary['train_loss']), '\t',
        'Validation Loss:',
            '{:.4e}'.format(epoch_summary['validation_loss']), '\t',
        'Time:', '{:.2f}'.format(epoch_summary['time']), 's',
    )