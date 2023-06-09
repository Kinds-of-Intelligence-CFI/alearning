import torch.nn as nn
import torch.nn.functional as F
import torch as th
import sys

KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
N_CHANNELS = 64
N_HIDDEN_FEATURES = 256
N_STIMULI = 20
# OUTPUT_SIZE = 20
N_ACTIONS = 7

class ALearningModel(nn.Module):
    def __init__(self, in_channels, in_width, in_height):
        super().__init__()

        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height

        self.visual_processor = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(N_CHANNELS, N_CHANNELS,
                      KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.AvgPool2d(KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        )

        self.softmax_layer = nn.Sequential(
            nn.Linear(N_HIDDEN_FEATURES, N_STIMULI),
            nn.LogSoftmax(dim=1)
        )
        # self.stimulus_output = nn.Sequential(
        #     nn.Linear(N_HIDDEN_FEATURES, OUTPUT_SIZE),
        #     nn.Sigmoid()
        # )

        self.lhs = nn.Sequential(
            # nn.Linear(N_STIMULI, N_STIMULI),
            # nn.ReLU(),
            # nn.Linear(OUTPUT_SIZE, OUTPUT_SIZE),
            # nn.ReLU(),
            nn.Linear(N_STIMULI, 1)
            # nn.Linear(OUTPUT_SIZE, 1)
        )

        self.rhs = nn.Sequential(
            # nn.Linear(N_STIMULI + N_ACTIONS, N_STIMULI + N_ACTIONS),
            # nn.ReLU(),
            # nn.Linear(OUTPUT_SIZE + N_ACTIONS, OUTPUT_SIZE + N_ACTIONS),
            # nn.ReLU(),
            nn.Linear(N_STIMULI + N_ACTIONS, 1)
            # nn.Linear(OUTPUT_SIZE + N_ACTIONS, 1)
        )

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
