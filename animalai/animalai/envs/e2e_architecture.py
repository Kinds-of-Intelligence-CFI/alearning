import torch.nn as nn
import torch.nn.functional as F
import torch as th
import sys

KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
DROPOUT = 0.1
N_CHANNELS = 512
N_HIDDEN_FEATURES = 2048
N_STIMULI = 30
# OUTPUT_SIZE = 128
N_ACTIONS = 7


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=KERNEL_SIZE,
                 stride=STRIDE, padding=PADDING, dropout=DROPOUT):
        super(ResBlock3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size, 1, padding)
        self.dropout = nn.Dropout3d(p=self.dropout)

        if out_channels != in_channels or stride > 1:
            self.conv_in = nn.Conv3d(in_channels, out_channels, 1, stride, 0)
        else:
            self.conv_in = None

    def forward(self, h):
        h = self.relu(self.bn1(h))
        out = self.relu(self.bn2(self.conv1(h)))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.conv_in is not None:
            residual = self.conv_in(h)
        else:
            residual = h

        out += residual
        return out

    def get_output_sizes(self, width, height):
        width = int((width + 2 * self.padding - self.kernel_size)
                    / self.stride + 1)
        height = int((height + 2 * self.padding - self.kernel_size)
                     / self.stride + 1)

        return width, height


class ALearningModel(nn.Module):
    def __init__(self, in_channels, in_width, in_height):
        super().__init__()

        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height

        # self.visual_processor = nn.Sequential(
        #     nn.BatchNorm3d(self.in_channels),
        #     nn.ReLU(),
        #     nn.Conv3d(self.in_channels, N_CHANNELS,
        #               KERNEL_SIZE, STRIDE, PADDING),
        #     nn.BatchNorm3d(N_CHANNELS),
        #     nn.ReLU(),
        #     nn.Conv3d(N_CHANNELS, N_CHANNELS,
        #               KERNEL_SIZE, STRIDE, PADDING),
        #     nn.BatchNorm3d(N_CHANNELS),
        #     nn.ReLU(),
        #     nn.Conv3d(N_CHANNELS, N_CHANNELS,
        #               KERNEL_SIZE, STRIDE, PADDING),
        #     nn.BatchNorm3d(N_CHANNELS),
        #     nn.ReLU(),
        #     nn.Conv3d(N_CHANNELS, N_CHANNELS,
        #               KERNEL_SIZE, STRIDE, PADDING),
        #     nn.BatchNorm3d(N_CHANNELS),
        #     nn.ReLU(),
        #     nn.Conv3d(N_CHANNELS, N_CHANNELS,
        #               KERNEL_SIZE, STRIDE, PADDING),
        #     nn.BatchNorm3d(N_CHANNELS),
        #     nn.ReLU(),
        #     nn.AvgPool3d((1, KERNEL_SIZE, KERNEL_SIZE),
        #                  stride=STRIDE, padding=(0, PADDING, PADDING))
        # )

        self.visual_processor = nn.Sequential(
            ResBlock3D(self.in_channels, N_CHANNELS),
            ResBlock3D(N_CHANNELS, N_CHANNELS),
            ResBlock3D(N_CHANNELS, N_CHANNELS),
            ResBlock3D(N_CHANNELS, N_CHANNELS),
            ResBlock3D(N_CHANNELS, N_CHANNELS),
            nn.BatchNorm3d(N_CHANNELS),
            nn.ReLU(),
            nn.AvgPool3d((1, KERNEL_SIZE, KERNEL_SIZE),
                         stride=STRIDE, padding=(0, PADDING, PADDING))
        )

        self.softmax_layer = nn.Sequential(
            nn.Linear(N_HIDDEN_FEATURES, N_STIMULI),
            nn.LogSoftmax(dim=1)
        )
        # self.stimulus_output = nn.Sequential(
        #     nn.Linear(N_HIDDEN_FEATURES, OUTPUT_SIZE),
        #     nn.Tanh()
        # )

        # self.lhs = nn.Sequential(
        #     # nn.Linear(N_STIMULI, N_STIMULI),
        #     # nn.ReLU(),
        #     # nn.Linear(OUTPUT_SIZE, OUTPUT_SIZE),
        #     # nn.ReLU(),
        #     nn.Linear(N_STIMULI, 1)
        #     # nn.Linear(OUTPUT_SIZE, 1)
        # )
        self.lhs = nn.Linear(N_STIMULI, 1)
        # self.lhs = nn.Linear(OUTPUT_SIZE, 1)

        # self.rhs = nn.Sequential(
        #     # nn.Linear(N_STIMULI + N_ACTIONS, N_STIMULI + N_ACTIONS),
        #     # nn.ReLU(),
        #     # nn.Linear(OUTPUT_SIZE + N_ACTIONS, OUTPUT_SIZE + N_ACTIONS),
        #     # nn.ReLU(),
        #     nn.Linear(N_STIMULI + N_ACTIONS, 1)
        #     # nn.Linear(OUTPUT_SIZE + N_ACTIONS, 1)
        # )
        self.rhs = nn.Linear(N_STIMULI + N_ACTIONS, 1)
        # self.rhs = nn.Linear(OUTPUT_SIZE + N_ACTIONS, 1)

    def forward(self, *args, **kwds):
        if len(args) == 1:
            img = args[0]
            encoded = self.visual_processor(img)
            encoded = th.reshape(encoded, (encoded.shape[0], -1))

            stimulus = F.gumbel_softmax(self.softmax_layer(encoded))
            # stimulus = self.stimulus_output(encoded)
            return stimulus
        elif len(args) == 0 and len(kwds) >= 1:
            stimulus = kwds["stimulus"]

            w_value = self.lhs(stimulus)
            sr_value = None
            if "onehot_action" in kwds:
                onehot_action = kwds["onehot_action"]
                cated = th.cat([stimulus, onehot_action], dim=1)
                sr_value = self.rhs(cated)

            return w_value, sr_value
        else:
            sys.stderr.write("incorrect arguments supplied to a-learning model")
            sys.exit(1)

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
