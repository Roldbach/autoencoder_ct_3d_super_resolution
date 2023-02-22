import torch
from torch import nn


class UNet(nn.Module):
    """U-Net model.

    This model replaces all 2D operations with their
    corresponding 3D versions without changing settings.
    Several modifications have been done to ensure a
    relatively fair comparison with other models.
    """
    def __init__(self) -> None:
        """Initialises UNet."""
        super().__init__()

        self._encoder = nn.ModuleDict({
            'level_1': DoubleConv3DBlock(1, 32),
            'level_2': nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1),
                DoubleConv3DBlock(32, 64),
            ),
            'level_3': nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1),
                DoubleConv3DBlock(64, 128),
            ),
            'level_4': nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1),
                DoubleConv3DBlock(128, 256),
            ),
        })
        self._decoder = nn.ModuleDict({
            'level_3_upsample': TrilinearInterpolationConv3DBlock(256, 128),
            'level_3': DoubleConv3DBlock(256, 128),
            'level_2_upsample': TrilinearInterpolationConv3DBlock(128, 64),
            'level_2': DoubleConv3DBlock(128, 64),
            'level_1_upsample': TrilinearInterpolationConv3DBlock(64, 32),
            'level_1': nn.Sequential(
                DoubleConv3DBlock(64, 32),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding='same'),
            ),
        })
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        encoder_output_level_1 = self._encoder['level_1'](input)
        encoder_output_level_2 = self._encoder['level_2'](encoder_output_level_1)
        encoder_output_level_3 = self._encoder['level_3'](encoder_output_level_2)
        encoder_output_level_4 = self._encoder['level_4'](encoder_output_level_3)

        decoder_output_level_3 = self._forward_decoder(
            encoder_output_level_4, encoder_output_level_3, 'level_3')
        decoder_output_level_2 = self._forward_decoder(
            decoder_output_level_3, encoder_output_level_2, 'level_2')
        decoder_output_level_1 = self._forward_decoder(
            decoder_output_level_2, encoder_output_level_1, 'level_1')

        return decoder_output_level_1
    
    def _forward_decoder(
        self, output: torch.Tensor, feature_copied: torch.Tensor, level: str,
    ) -> torch.Tensor:
        """Upsamples, concatenates and returns output from the decoder."""
        output = self._decoder[f'{level}_upsample'](output)
        output = torch.concat((output, feature_copied), axis=1)
        output = self._decoder[level](output)

        return output

class Conv3DBlock(nn.Module):
    """Conv3D + LeakyReLU (negative_slope=0.1)"""
    def __init__(self, channel_in: int, channel_out: int) -> None:
        """Initialises Conv3DBlock."""
        super().__init__()

        self._layer = nn.Sequential(
            nn.Conv3d(
                channel_in, channel_out, kernel_size=3, stride=1, padding='same',
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        return self._layer(input)

class DoubleConv3DBlock(nn.Module):
    """[Conv3D + LeakyReLU (negative_slope=0.1)] * 2"""
    def __init__(self, channel_in: int, channel_out: int) -> None:
        """Initialises DoubleConv3DBlock."""
        super().__init__()

        self._layer = nn.Sequential(
            Conv3DBlock(channel_in, channel_out),
            Conv3DBlock(channel_out, channel_out),
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