"""Script for model testing.

This is the script used for model training. To test a
model, please specify the following experiment settings:
    (1) model_name: str, one of PlainCNN/AE_Maxpool/AE_Conv/
        UNet.
    (2) upsample_name: str, one of trilinear_interpolation/
        same_insertion.
    (3) weight_file_path: str, the file that stores
        model weights.
    (4) scale_factor: int, the scale factor of downsampling
        /upsampling in the z-axis.
    (5) window: tuple[float|None, float|None], the range
        of HU values interested.

This script can be interacted directly from the terminal.
"""
import argparse
import sys

from helper import test_helper


def main() -> int:
    test_helper.test(parse_argument())

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
        '--weight_file_path',
        type = str,
        required = True,
        help = 'The file that stores model weights',
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