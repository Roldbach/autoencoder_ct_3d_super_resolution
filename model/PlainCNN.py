import torch
from torch import nn


# ----- Building Block -----
class Conv3DBlock(nn.Module):
    """Conv3D + LeakyReLU(negative_slope=0.1)"""
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
        """Returns the output by forward calculation."""
        return self._layer(input)


# ----- Model -----
class PlainCNN(nn.Module):
    """Plain CNN model.

    This model consists of 12 basic building blocks, which
    is Conv3D + LeakyReLU(negative_slope=0.1), in series.
    The global residual learning is also applied.
    """
    def __init__(self) -> None:
        """Initialises PlainCNN."""
        super(PlainCNN, self).__init__()

        self._block_in = Conv3DBlock(1, 64)
        self._block_middle_iter = nn.ModuleList(Conv3DBlock(64, 64) for i in range(10))
        self._block_out = Conv3DBlock(64, 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns the output by forward calculation."""
        residual = input.clone()

        for block in [self._block_in, *self._block_middle_iter, self._block_out]:
            residual = block(residual)
        
        return input + residual