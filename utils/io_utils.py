"""Utils for data input/output.

This module contains utility functions for data
input/output. Wrapper functions are provided so the correct
utility function could be chosen based on the file name
extension.
"""
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from torch import nn


def read_dicom_image_directory(directory_path: str) -> np.ndarray:
    """Reads a 3D Dicom image from the given directory.

    Reads a 3D Dicom image from the given directory using
    SimpleITK API. All slices are read in order.

    Args:
        directory_path:
            A str that specifies the path of the directory
            that contains all Dicom files that belongs to
            the same 3D image.
    
    Returns:
        A numpy.ndarray that contains all HU values of the
        3D image.    
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(directory_path))

    return sitk.GetArrayFromImage(reader.Execute())
    
def read_file(file_path: str) -> tuple[str, ...] | pd.DataFrame:
    """Reads a file and returns its content.

    Reads a file and returns its content. The corresponding
    utility function is chosen based on the file name
    extension. This utility function supports the following
    files:
        (1) .csv
        (2) .txt where content are separated line-by-line
    
    Args:
        file_path:
            A str that specifies the path of the file.
    
    Returns:
        A tuple[str, ...] | pd.Dataframe that contains the
        content in the file.
    """
    if file_path.endswith('.csv'):
        return _read_csv_file(file_path)
    elif file_path.endswith('.txt'):
        return _read_txt_file(file_path)

def _read_csv_file(file_path: str) -> pd.DataFrame:
    """Reads a .csv file."""
    return pd.read_csv(file_path)

def _read_txt_file(file_path: str) -> tuple[str, ...]:
    """Reads a .txt file line-by-line."""
    with open(file_path, 'r') as file:
        line_all = file.readlines()
    
    return tuple(line.strip() for line in line_all)

def read_weight(file_path: str, model: nn.Module) -> nn.Module:
    """Reads and loads weights from the path into a model.

    Reads and loads weights from the path into a model.
    This utility functions could be used when arbitrary
    number of GPUs are available.

    Args:
        file_path:
            A str that specifies the path of the file.
        model:
            A torch.nn.Module that specifies the
            architecture of the model.
    
    Returns:
        A torch.nn.Module in which each layer has loaded
        the corresponding weights.
    """
    try:
        model.load_state_dict(torch.load(file_path))
    except:
        model.module.load_state_dict(torch.load(file_path))

    return model

def write_file(file_path: str, content: pd.DataFrame) -> None:
    """Writes the content to the given file path.

    Writes the content to the given file path. The
    corresponding utility function is chosen based on the
    file name extension. This utility function supports the
    following file:
        (1) .csv
    
    Args:
        file_path:
            A str that specifies the path of the file.
        content:
            A pd.Dataframe that contains data to be stored.    
    """
    if file_path.endswith('.csv'):
        _write_csv_file(file_path, content)

def _write_csv_file(file_path: str, content: pd.DataFrame) -> None:
    """Writes the content as .csv file to the given path."""
    content.to_csv(file_path, index=False)

def write_weight(file_path: str, model: nn.Module) -> None:
    """Writes weights from the model to the given path.

    Writes weights from the model to the given path. This
    utility function could be used when arbitrary number of
    GPUs are available.

    Args:
        file_path:
            A str that specifies the path of the file.    
        model:
            A torch.nn.Module in which each layer has its
            weights.
    """
    try:
        torch.save(model.module.state_dict(), file_path)
    except:
        torch.save(model.state_dict(), file_path)