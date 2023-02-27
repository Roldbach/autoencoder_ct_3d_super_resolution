"""Device-related configurations.

This module contains configurations about the devices used
in the experiment. All devices are torch-based and multiple
GPUs could also be configured if available.
"""
import torch


# Device used in model training
TRAIN_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Device ID used in model training
# None if 0/1 GPU is available, otherwise list all device IDs
TRAIN_DEVICE_ID = (0, 1)

# Device used in model inference
TEST_DEVICE = torch.device('cpu')
# Device ID used in model inference
# None if 0/1 GPU is available, otherwise list all device IDs
TEST_DEVICE_ID = None