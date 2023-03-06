"""Script for model testing.
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
        '--scale_factor',
        type = int,
        required = True,
        help = 'The scale factor of downsampling/upsampling in the z-axis',
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