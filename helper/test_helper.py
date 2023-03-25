"""Helper of model testing.

This module contains class and functions that can
facilitate the model testing. The main function test() is
compatible with arguments from the terminal. 
"""
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from configuration import device_configuration
from dataset.template_dataset import TemplateDataset
from model.Autoencoder import AutoencoderConv
from model.Autoencoder import AutoencoderMaxpool
from model.PlainCNN import PlainCNN
from model.UNet import UNet
from utils import evaluation_utils
from utils import image_utils
from utils import io_utils


class TestDelegate:
    """A delegate designed for model testing only.

    A delegate that is specifically designed for model
    testing. It contains all components required and 
    some convenient functions that fascilitate testing
    experiments. Arguments are directly obtained from the
    terminal.
    """

    def __init__(self, argument: Namespace) -> None:
        """Initialises TestDelegate."""
        self._test_data_loader = self._construct_test_data_loader()
        self._model = self._construct_model(argument)
    
    def _construct_test_data_loader(self) -> DataLoader:
        """Constructs a dataloader to load test data."""
        return DataLoader(
            dataset = TemplateDataset('test'),
            batch_size = 1,
            shuffle = False,
            num_workers = 4,
        )

    def _construct_model(self, argument: Namespace) -> nn.Module:
        """Constructs a model used in the experiment.

        Constructs a model based on the given model name.
        The model is moved to the test device and converted
        to the data parallelism if multiple GPUs are
        available. Model weights are also loaded.

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
        device = device_configuration.TEST_DEVICE
        device_id = device_configuration.TEST_DEVICE_ID

        match argument.model_name:
            case 'PlainCNN':
                model = PlainCNN().to(device)
            case 'AE_Maxpool':
                model = AutoencoderMaxpool().to(device)
            case 'AE_Conv':
                model = AutoencoderConv().to(device)
            case 'UNet':
                model = UNet().to(device)
            case _:
                raise ValueError(
                    f'Can not find model: {argument.model_name}')

        if device_id is not None:
            model = nn.parallel.DataParallel(model, device_id)
        
        model = io_utils.read_weight(argument.weight_file_path, model)
        
        return model

def test(argument: Namespace) -> None:
    """Tests the model using given experiment settings.

    Tests the model using given experiment settings by the
    following steps:
        (1) Generate a prediction for every input in the
            test dataset.
        (2) Computes candidate (prediction vs label) and
            reference (input vs label) evaluation results.
        (3) Repeat step 1~2 for all input in the test
            dataset.
        (4) Report mean and standard deviation for all
            evaluation results in the terminal.
        
    Args:
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal
    """
    delegate = TestDelegate(argument)
    evaluation_metric_all = ('PSNR', 'SSIM', "RMSE")
    candidate_evaluation_result_map = defaultdict(list)
    reference_evaluation_result_map = defaultdict(list)

    delegate._model.eval()
    for input, label in delegate._test_data_loader:
        input = pre_process_input(input, argument)
        label = pre_process_label(label, argument)

        prediction = delegate._model(
            input.to(device_configuration.TEST_DEVICE))
        
        input = post_process_tensor(input)
        prediction = post_process_tensor(prediction)
        label = post_process_tensor(label)

        for evaluation_metric in evaluation_metric_all:
            candidate_evaluation_result_map[evaluation_metric].append(
                evaluate(prediction, label, evaluation_metric))
            reference_evaluation_result_map[evaluation_metric].append(
                evaluate(input, label, evaluation_metric))
    
    report_evaluation_result(
        candidate_evaluation_result_map, reference_evaluation_result_map)

def pre_process_input(
    input: torch.Tensor, argument: Namespace) -> torch.Tensor:
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
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal.  

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed input.
    """
    input = image_utils.normalise_pixel(input, *argument.window)
    input = image_utils.truncate_image(input, 8)
    input = image_utils.downsample_image_x_y_axis(input)
    input = image_utils.downsample_image_z_axis(input, argument.scale_factor)
    input = image_utils.upsample_image_z_axis(
        input, argument.scale_factor, argument.upsample_name)

    return input

def pre_process_label(
    label: torch.Tensor, argument: Namespace) -> torch.Tensor:
    """Pre-processes label.

    Pre-processes label before model testing with the following steps:
        (1) Normalises the input with the given window.
        (2) Truncates the input so its shape can fit
            non-overlapping patching.
        (3) Downsamples the input in x, y axes by x2.
    
    Args:
        label:
            A torch.Tensor that contains pixel values of
            the label.
        argument:
            A argparse.Namespace that contains arguments
            directly from the terminal.

    Returns:
        A torch.Tensor that contains pixel values of
        the pre-processed label.
    """
    label = image_utils.normalise_pixel(label, *argument.window)
    label = image_utils.truncate_image(label, 8)
    label = image_utils.downsample_image_x_y_axis(label)

    return label

def post_process_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Post-processes tensor.
    
    Post-processes tensor with the following steps:
        (1) Recovers pixel values to [0, 255] and converts the data type to
            uint8.
        (2) Converts tensor to array.
    """
    output = image_utils.recover_pixel(tensor, 0, 255, torch.uint8)
    output = image_utils.convert_tensor_to_array(output)

    return output

def evaluate(
    candidate: np.ndarray,
    reference: np.ndarray,
    evaluation_metric: str,
    data_range: int = 255,
) -> float:
    """Evaluates candidate and reference using the metric.
    
    Evaluates candidate and reference using the given
    evaluation metric.

    Args:
        candidate:
            A numpy.ndarray that contains all pixel values
            of the candidate.
        reference:
            A numpy.ndarray that contains all pixel values
            of the reference.
        evaluation_metric:
            A str that specifies the name of evaluation metric.
        data_range:
            An int that specifies the range of pixel values
            within images. Both images should share the 
            same data range.
    
    Returns:
        A float that represents the evaluation results
        between the candidate and the reference.
    
    Raises:
        ValueError: The given evaluation metric is invalid.
    """
    match evaluation_metric:
        case 'PSNR':
            return evaluation_utils.evaluate_PSNR(
                candidate, reference, data_range)
        case 'SSIM':
            return evaluation_utils.evaluate_SSIM(
                candidate, reference, data_range)
        case 'RMSE':
            return evaluation_utils.evaluate_RMSE(candidate, reference)
        case _:
            raise ValueError(
                f'Can not find evaluation metric: {evaluation_metric}')

def report_evaluation_result(
    candidate_evaluation_result_map: dict[str, list[float]],
    reference_evaluation_result_map: dict[str, list[float]],
) -> None:
    """Reports evaluation results in the terminal."""
    for evaluation_metric in candidate_evaluation_result_map:
        print(
            '{} Mean: {:.4f} / {:.4f}'.format(
                evaluation_metric,
                np.mean(candidate_evaluation_result_map[evaluation_metric]),    
                np.mean(reference_evaluation_result_map[evaluation_metric]),
            )
        )
    
        print(
            '{} STD: {:.4f} / {:.4f}'.format(
                evaluation_metric,
                np.std(candidate_evaluation_result_map[evaluation_metric]),
                np.std(reference_evaluation_result_map[evaluation_metric]),
            )
        )

        print('')