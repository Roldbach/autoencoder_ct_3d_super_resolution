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
    (10) window: tuple[int|None, int|None], the range of HU
        values interested.

This script can be interacted directly from the terminal.
"""
import argparse
import sys

from helper import train_helper


def main() -> int:
    """Runs the main script."""
    train_helper.train(parse_argument())

    return 0

def parse_argument() -> argparse.Namespace:
    """Parse arguments from the terminal."""
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument(
        '--model_name',
        type = str,
        required = True,
        help = 'The name of the model: PlainCNN | AE_Maxpool | AE_Conv | UNet',
    )
    parser.add_argument(
        '--upsample_name',
        type = str,
        required = True,
        help = 'The name of the upsampling method: trilinear_interpolation | same_insertion',
    )
    parser.add_argument(
        '--output_directory_path',
        type = str,
        required = True,
        help = 'The directory to store model weights and record',
    )
    parser.add_argument(
        '--is_resuming',
        action = 'store_true',
        help = 'Whether to resume from the last experiment',
    )
    parser.add_argument(
        '--epoch',
        type = int,
        required = True,
        help = 'The number of epochs',
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        required = True,
        help = 'The number of data in one batch',
    )
    parser.add_argument(
        '--patch_size',
        type = int,
        required = True,
        help = 'The size of cubic patches',
    )
    parser.add_argument(
        '--scale_factor',
        type = int,
        required = True,
        help = 'The scale factor of downsampling/upsampling in the z-axis',
    )
    parser.add_argument(
        '--learning_rate',
        type = float,
        required = True,
        help = 'The step size in gradient updating',
    )
    parser.add_argument(
        '--window',
        type = int,
        nargs = '+',
        default = (None, None),
        help = 'The range of HU values interested'
    )

    return parser.parse_args()

if __name__ == '__main__':
    sys.exit(main())