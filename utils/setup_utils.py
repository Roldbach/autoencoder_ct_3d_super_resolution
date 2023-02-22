"""Utils for experiment setup.

This module contains utility functions for experiment
setup, which includes:
    (1) data_loader
    (2) model
    (3) optimiser
    (4) loss function
    (5) record of the experiment
"""
from typing import Iterable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.AapmMayoDataset import AapmMayoDataset
from model.Autoencoder import Autoencoder_Conv
from model.Autoencoder import Autoencoder_Maxpool
from model.PlainCNN import PlainCNN
from model.UNet import UNet


def set_up_data_loader(tag: str, shuffle: bool = True) -> DataLoader:
    """Sets up a data loader.
    
    Sets up and returns a data loader that can be used in
    the experiment. The loaded data depends on the given
    tag. The data loader should be able to yield raw 3D
    images (5D tensors) one-by-one.

    Args:
        tag:
            A str that specifies which part of data within
            the dataset to be loaded.    
        shuffle:
            A bool that specifies whether to shuffle the
            order of data.
    
    Returns:
        A torch.utils.data.DataLoader that should be able
        to yield raw 3D images (5D tensors) one-by-one.
    """
    return DataLoader(
        dataset = AapmMayoDataset(tag),
        batch_size = 1,
        shuffle = shuffle,
        num_workers = 4,
    )

def set_up_model(
    model_name: str,
    device: torch.device,
    device_id: tuple[int, ...] | None = None,
) -> nn.Module:
    """Sets up a model.

    Sets up and returns a model based on the given model
    name. The model is moved to the given device and also
    coverted to the data parallelism if multiple GPUs are
    available.

    Args:
        model_name:
            A str that specifies the name of the model.    
        device:
            A torch.device that specifies the device used
            in the experiment.
        device_id:
            A tuple[int, ...] | None that specifies indices
            of all GPUs if multiple GPUs are available.
        
    Returns:
        A torch.nn.Module that specifies the corresponding
        architecture of the model.
    
    Raises:
        ValueError: The given model_namee is invalid.    
    """
    match model_name:
        case 'PlainCNN':
            model = PlainCNN().to(device)
        case 'AE_Maxpool':
            model = Autoencoder_Maxpool().to(device)
        case 'AE_Conv':
            model = Autoencoder_Conv().to(device)
        case 'UNet':
            model = UNet().to(device)
        case _:
            raise ValueError(f'Can not find model: {model_name}')

    if device_id is not None:
        model = nn.parallel.DataParallel(model, device_id)
    
    return model

def set_up_optimiser(
    parameter: Iterable,
    learning_rate: float,
) -> torch.optim.Adam:
    """Sets up an Adam optimiser."""
    return torch.optim.Adam(parameter, learning_rate, betas=(0.9, 0.99))

def set_up_loss_function() -> nn.MSELoss:
    """Sets up MSE (L2) as loss function."""
    return nn.MSELoss()

def set_up_record() -> pd.DataFrame:
    """Sets up an empty record."""
    return pd.DataFrame()