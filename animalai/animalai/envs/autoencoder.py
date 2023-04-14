import torch.nn as nn
import torch as th

KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
N_CHANNELS = 64


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, in_width, in_height):
        super().__init__()

        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(N_CHANNELS, N_CHANNELS,
                               KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.ConvTranspose2d(N_CHANNELS, N_CHANNELS,
                               KERNEL_SIZE, STRIDE, PADDING,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(N_CHANNELS, N_CHANNELS,
                               KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.ConvTranspose2d(N_CHANNELS, N_CHANNELS,
                               KERNEL_SIZE, STRIDE, PADDING),
            nn.ReLU(),
            nn.ConvTranspose2d(N_CHANNELS, N_CHANNELS,
                               KERNEL_SIZE, STRIDE, PADDING,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(N_CHANNELS, self.in_channels,
                               KERNEL_SIZE, STRIDE, PADDING,
                               output_padding=1)
        )

    def forward(self, img):
        encoded = self.encoder(img)
        reconstructed = self.decoder(encoded)

        encoded = th.reshape(encoded, (encoded.shape[0], -1))

        return encoded, reconstructed

    @staticmethod
    def _update_width_height(width, height, kernel_size=KERNEL_SIZE,
                             stride=STRIDE, padding=PADDING):
        width = int((width + 2 * padding - kernel_size)
                    / stride + 1)
        height = int((height + 2 * padding - kernel_size)
                     / stride + 1)

        return width, height

    @staticmethod
    def _update_width_height_transposed(width, height,
                                        kernel_size=KERNEL_SIZE,
                                        stride=STRIDE, padding=PADDING,
                                        output_padding=0):
        width = (width - 1) * stride - 2 * padding \
            + kernel_size + output_padding
        height = (height - 1) * stride - 2 * padding \
            + kernel_size + output_padding

        return width, height
