from animalai.envs.e2e_architecture import ALearningModel
from animalai.envs.stimulus_e2e import StimulusE2E
from animalai.envs.e2e_dataset import E2EDataset
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.special import softmax
import random
from itertools import groupby
import os


def to_tensor(array):
    res = array.copy()
    return th.from_numpy(res) / 255

def normalise(img, mean, std):
    mean = th.tensor(mean).reshape(3, 1, 1, 1)
    std = th.tensor(std).reshape(3, 1, 1, 1)
    res = (img - mean) / std
    return res


class Normalise(nn.Module):
    def __init__(self, mean, std):
        super(Normalise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        return normalise(img, self.mean, self.std)

class ALearnerE2E():
    """Implements the A-learning algorithm
    Can change the number of rays but only responds to GOODGOALs, GOODGOALMULTI and BADGOAL"""

    def __init__(self, n_actions, in_channels,
                 in_width, in_height, gpu=True,
                 temperature=100,
                 discount=0.7,
                 model_file=None):
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height

        self.temperature = temperature
        self.initial_temperature = temperature
        self.discount = discount

        self.w_values = defaultdict(float)
        self.sr_values = defaultdict(float)

        self.n_actions = n_actions

        self.n_epochs = 10
        self.use_target_value = True

        self.gpu = gpu
        self.aler = ALearningModel(in_channels, in_width, in_height)
        if self.gpu:
            self.aler = self.aler.to(0)

        self.model_file = model_file
        if os.path.exists(self.model_file):
            if gpu:
                self.aler.load_state_dict(th.load(model_file))
            else:
                self.aler.load_state_dict(
                    th.load(model_file,
                            map_location=th.device('cpu')
                            ))

        self.optimiser = th.optim.Adam(self.aler.parameters(), lr=0.001,
                                       weight_decay=1e-5)
        # self.optimiser = th.optim.SGD(self.aler.parameters(), lr=0.01,
        #                               momentum=0.9, nesterov=True)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.MSELoss(reduction='none')

    def reset_optimiser(self):
        self.optimiser = th.optim.Adam(self.aler.parameters(), lr=0.001,
                                       weight_decay=1e-5)
        # self.optimiser = th.optim.SGD(self.aler.parameters(), lr=0.01,
        #                               momentum=0.9, nesterov=True)
        self.n_epochs = 10

    def set_target_value(self):
        self.use_target_value = True

    def get_stimulus(self, obs):
        return self.aler(obs)

    def get_action(self, obs, reward=None) -> int:
        """Returns the action to take given the current observation"""
        with th.no_grad():
            stim = self.aler(obs)
            stim = StimulusE2E(stim, reward=reward)
            output = self.aler(stimulus=stim.stimulus)
            self.w_values[stim] = output[0].item()
            for a in range(self.n_actions):
                self.sr_values[(stim, a)] = output[a+1].item()
            # self.w_values[stim] = max([self.sr_values[(stim, a)]
            #                            for a in range(self.n_actions)
            #                            ])

        all_keys = list(map(
            lambda a: (stim, a),
            range(self.n_actions)
        ))

        all_sr_values = np.fromiter(
            map(lambda k: self.sr_values[k], all_keys),
            dtype=float
        )
        probs = softmax(all_sr_values / self.temperature)
        draw = random.random()
        action = 0
        cum_prob = 0
        for prob in probs:
            cum_prob += prob
            if draw <= cum_prob:
                break
            # this checks the edge case when there are rounding errors
            if action < self.n_actions - 1:
                action += 1
        # max_idx = np.argmax(all_sr_values)
        # action = all_keys[max_idx][1]

        return stim, action

    def do_training_round(self, data):
        if self.use_target_value:
            aler = ALearningModel(self.in_channels,
                                  self.in_width,
                                  self.in_height)
            if self.gpu:
                aler = aler.to(0)
            aler.load_state_dict(self.aler.state_dict())
        else:
            aler = self.aler
        dataset = E2EDataset(data, aler, self.n_actions, gpu=self.gpu,
                             train_transform=transforms.Compose([
                                 to_tensor,
                                 transforms.RandomCrop(84, padding=10,
                                                       padding_mode='reflect'),
                                 transforms.RandomVerticalFlip(),
                                 Normalise((0.6282, 0.6240, 0.5943),
                                           (0.1751, 0.1605, 0.2117))
                             ]),
                             test_transform=transforms.Compose([
                                 to_tensor,
                                 Normalise((0.6282, 0.6240, 0.5943),
                                           (0.1751, 0.1605, 0.2117))
                             ]))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        print("\ndoing training round")
        for i in range(self.n_epochs):
            total_loss = 0
            steps = 0
            for (imgs, actions, w_vals, u_vals,
                 weights, W_vals, U_vals) in iter(loader):
            # for imgs, actions, w_vals, u_vals in iter(loader):
                stimuli = self.aler(imgs)
                output = self.aler(stimulus=stimuli)
                w_values = output[:, 0]
                sr_values = th.gather(output, 1, (actions+1))

                l1 = th.mean(
                    weights * self.criterion(w_values,
                                             self.discount *
                                             th.max(w_vals + u_vals,
                                                    W_vals + U_vals))
                )

                l2 = th.mean(
                    weights * self.criterion(sr_values,
                                             self.discount *
                                             th.max(w_vals + u_vals,
                                                    W_vals + U_vals))
                )
                loss = (l1 + l2) / 2

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                total_loss += loss.item()
                steps += 1
            print("epoch %d | loss = %.4e" % (i+1, total_loss / steps))
        print("\n")

    def decrease_temperature(self):
        if self.temperature > 10:
            self.temperature -= 10
        else:
            self.temperature = 1

    def exploit(self):
        self.temperature = 1

    def reset_temperature(self):
        self.temperature = self.initial_temperature

    def print_max_stim_val(self):
        max_stim_value = max(self.w_values.values())
        max_sr_value = max(self.sr_values.values())
        print("Max stimulus value: %.4f" % max_stim_value)
        print("Max S-R value: %.4f" % max_sr_value)

    def save_model(self):
        th.save(self.aler.state_dict(), self.model_file)
