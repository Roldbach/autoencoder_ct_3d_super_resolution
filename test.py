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
"""
from helper import test_helper


if __name__ == '__main__':
    # Clean this
    model_name = 'PlainCNN'
    upsample_name = 'trilinear_interpolation'
    scale_factor = 8
    weight_path = f'./weight/{model_name}_{upsample_name}_x{scale_factor}.pth'
    window = (-1024, 1476)

    delegate = test_helper.TestDelegate(
        model_name = model_name,
        upsample_name = upsample_name,
        scale_factor = scale_factor,
        weight_file_path = weight_path,
        window = (-1024, 1476),
    )

    test_helper.test(delegate)