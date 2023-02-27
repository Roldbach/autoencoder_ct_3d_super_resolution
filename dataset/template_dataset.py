import numpy as np
import torch
from torch.utils.data import Dataset

from configuration import path_configuration
from utils import io_utils


class TemplateDataset(Dataset):
    """A template dataset used in all experiments.

    A template dataset that can be used in all experiments.
    To use this, please provide a .csv file that contains
    all paths to your image data and their tags. The loading
    file should contain the following columns:
        (1) path: A str that specifies the path of 3D image
            This could be a .nii/.npy/dicom directory
        (2) tag: One from train/validation/test
    
    We have provided a sample loading file as demonstration.
    """

    def __init__(
        self,
        tag: str,
        loading_file_path: str =
            path_configuration.DATASET_LOADING_FILE_PATH,
    ) -> None:
        """Initialises TemplateDataset."""
        self._image_path_all = io_utils.read_file(loading_file_path)
        self._image_path_all = self._image_path_all.loc[
            self._image_path_all['tag'] == tag]
        self._image_path_all = self._image_path_all['path'].tolist()
    
    def __len__(self) -> int:
        return len(self._image_path_all)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = io_utils.read_3D_image(self._image_path_all[index])
        image = self._convert_array_to_tensor(image)

        return image, image

    def _convert_array_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = np.expand_dims(image, axis=0)

        return torch.from_numpy(image)