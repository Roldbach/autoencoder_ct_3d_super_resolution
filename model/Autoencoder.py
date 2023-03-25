import torch
from torch import nn


class AutoencoderMaxpool(nn.Module):
    """AE-Maxpool model.

    This model is built upon the baseline model(Plain CNN).
    Max-pooling is utilized as the downsampling method and
    'trilinear interpolation + convolution' is used as the
    upsampling method. The global residual learning is also
    applied.
    """
    def __init__(self) -> None:
        """Initialises Autoencoder_Maxpool."""
        super().__init__()

        self._encoder = nn.Sequential(
            Conv3DBlock(1, 64),
            Conv3DBlock(64, 64),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1),
            Conv3DBlock(64, 128),
            Conv3DBlock(128, 128),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1),
            Conv3DBlock(128, 256),
            Conv3DBlock(256, 256),
        )

        self._decoder = nn.Sequential(
            Conv3DBlock(256, 256),
            Conv3DBlock(256, 128),
            TrilinearInterpolationConv3DBlock(128, 128),
            Conv3DBlock(128, 128),
            Conv3DBlock(128, 64),
            TrilinearInterpolationConv3DBlock(64, 64),
            Conv3DBlock(64, 64),
            Conv3DBlock(64, 1),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        residual = input

        residual = self._encoder(residual)
        residual = self._decoder(residual)

        return input + residual

class AutoencoderConv(nn.Module):
    """AE-Conv model.

    This model is built upon the baseline model(Plain CNN).
    Stride convolution is served as the downsampling method
    and 'trilinear interpolation + convolution' is used as
    the upsampling method. The global residual learning
    is also applied.
    """
    def __init__(self) -> None:
        """Initialises Autoencoder_Conv."""
        super().__init__()

        self._encoder = nn.Sequential(
            Conv3DBlock(1, 64),
            Conv3DBlock(64, 64),
            StrideConv3DBlock(64, 64),
            Conv3DBlock(64, 128),
            Conv3DBlock(128, 128),
            StrideConv3DBlock(128, 128),
            Conv3DBlock(128, 256),
            Conv3DBlock(256, 256),
        )

        self._decoder = nn.Sequential(
            Conv3DBlock(256, 256),
            Conv3DBlock(256, 128),
            TrilinearInterpolationConv3DBlock(128, 128),
            Conv3DBlock(128, 128),
            Conv3DBlock(128, 64),
            TrilinearInterpolationConv3DBlock(64, 64),
            Conv3DBlock(64, 64),
            Conv3DBlock(64, 1),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        residual = input

        residual = self._encoder(residual)
        residual = self._decoder(residual)

        return input + residual

class Conv3DBlock(nn.Module):
    """Conv3D + LeakyReLU (negative_slope=0.1)"""
    def __init__(self, channel_in: int, channel_out: int) -> None:
        """Initialises Conv3DBlock."""
        super().__init__()

        self._layer = nn.Sequential(
            nn.Conv3d(
                channel_in, channel_out, kernel_size=3, stride=1, padding='same',
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        return self._layer(input)

class StrideConv3DBlock(nn.Module):
    """Conv3D (stride=2) + LeakyReLU (negative_slope=0.1)"""
    def __init__(self, channel_in: int, channel_out: int) -> None:
        """Initialises StrideConv3DBlock."""
        super().__init__()

        self._layer = nn.Sequential(
            nn.Conv3d(
                channel_in, channel_out, kernel_size=3, stride=2, padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        return self._layer(input)

class TrilinearInterpolationConv3DBlock(nn.Module):
    """Trilinear Interpolation (scale_factor=2) + Conv3D (kernel_size=1)"""
    def __init__(self, channel_in: int, channel_out: int) -> None:
        """Initialises TrilinearInterpolationLayer."""
        super().__init__()

        self._layer = nn.Conv3d(
            channel_in, channel_out, kernel_size=1, stride=1, padding='same')
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        return self._layer(
            nn.functional.interpolate(input, scale_factor=2, mode='trilinear'))