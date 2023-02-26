"""Script for model training.

This is the script used for model training. To train a
model, please specify the following experiment settings:
    (1) model_name: str, one of PlainCNN/AE_Maxpool/AE_Conv/
        UNet.
    (2) upsample_name: str, one of trilinear_interpolation/
        same_insertion.
    (3) output_directory_path: str, the directory to store
        model weights and record.
    (4) is_resuming: bool, whether resuming from the last
        experiment.
    (5) epoch: int, the number of epochs.
    (6) batch_size: int, the number of data in one batch.
    (7) patch_size: int, the number of cubic patches.
    (8) scale_factor: int, the scale factor of downsampling
        /upsampling in the z-axis.
    (9) learning_rate: float, the step size in gradient
        updating.
    (10) window: tuple[float|None, float|None], the range
        of HU values interested.
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