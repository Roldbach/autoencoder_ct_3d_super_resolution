"""Script for model training.

This is the main script used for model training.
"""
from helper import train_helper


if __name__ == '__main__':
    delegate = train_helper.TrainDelegate(
        model_name = 'PlainCNN',
        upsample_name = 'same_insertion',
        scale_factor = 4,
        is_resuming = True,
        output_directory_path = './temp',
        epoch = 500,
        batch_size = 16,
        patch_size = 64,
        learning_rate = 1e-5,
        window = (-1024, 1476),
    )

    train_helper.train(delegate)