"""Path-related configurations.

This module contains configurations about the paths used
in the experiment. 
"""
import os


# Path of the project root directory
ROOT_DIRECTORY_PATH = os.getcwd()
# Path of the dataset loading file
DATASET_LOADING_FILE_PATH = ''
# Path of the directory storing all pre-trained model weigths
WEIGHT_DIRECTORY_PATH = f'{ROOT_DIRECTORY_PATH}/weight'