"""Device-related configurations.

This module contains configurations about the devices used
in the experiment. All devices are torch-based and multiple
GPUs could be configured if available.
"""
import torch


# Device used in model training
TRAIN_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Device ID used in model training
TRAIN_DEVICE_ID = [i for i in range(torch.cuda.device_count())]

# Device used in model inference
TEST_DEVICE = torch.device('cpu')
# Device ID used in model inference
TEST_DEVICE_ID = []