from animalai.envs.e2e_architecture import ALearningModel
from collections import defaultdict
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import random
from itertools import groupby


class ALearnerE2E():
    """Implements the A-learning algorithm
    Can change the number of rays but only responds to GOODGOALs, GOODGOALMULTI and BADGOAL"""

    def __init__(self, n_actions, in_channels,
                 in_width, in_height, gpu=True, temperature=100):
        self.temperature = temperature
        self.initial_temperature = temperature

        self.w_values = defaultdict(float)
        self.sr_values = defaultdict(float)

        self.n_actions = n_actions
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height

        self.aler = ALearningModel(in_channels, in_width, in_height)
        self.optimiser = th.optim.SGD(self.aler.parameters(), lr=0.01,
                                      momentum=0.9,
                                      nesterov=True)
        # self.optimiser = th.optim.Adam(self.aler.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.gpu = gpu

        # list of stimulus-action pairs
        self.trajectory = []
        self.prev_stim = None

    def get_stimulus(self, obs):
        return self.aler(obs)

    def get_action(self, stimulus) -> int:
        """Returns the action to take given the current observation"""
        self.prev_stim = stimulus
        with th.no_grad():
            w_value, _ = self.aler(stimulus=stimulus.stimulus)
            self.w_values[stimulus] = w_value.item()
            for a in range(self.n_actions):
                onehot_action = F.one_hot(th.tensor([a]),
                                          num_classes=self.n_actions)
                if self.gpu:
                    onehot_action = onehot_action.to(0)
                _, sr_value = self.aler(stimulus=stimulus.stimulus,
                                        onehot_action=onehot_action)
                self.sr_values[(stimulus, a)] = sr_value.item()

        all_keys = list(map(
            lambda a: (self.prev_stim, a),
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

        self.trajectory.append((self.prev_stim, action))
        return action

    def update_stimulus_values(self, final_stim):
        groups = []

        for k, group in groupby(self.trajectory,
                                key=lambda x: hash(tuple(
                                    # x[0].onehot.detach().clone()[0].tolist()
                                    x[0].onehot.tolist()
                                ))):
            group = list(group)
            group.reverse()
            groups.append(group[0])

        groups.reverse()

        mean_loss = None
        target_value = th.tensor([[final_stim.u_val]]).float()

        target_aler = ALearningModel(self.in_channels,
                                     self.in_width,
                                     self.in_height)
        if self.gpu:
            target_aler = target_aler.to(0)
        target_aler.load_state_dict(self.aler.state_dict())
        if self.gpu:
            target_value = target_value.to(0)
        for stim, action in groups:
            onehot_action = F.one_hot(th.tensor([action]),
                                      num_classes=self.n_actions)

            w_value, sr_value = self.aler(stimulus=stim.stimulus,
                                          onehot_action=onehot_action)
            self.w_values[stim] = w_value.item()
            l1 = self.criterion(w_value, target_value)

            self.sr_values[(stim, action)] = sr_value.item()
            l2 = self.criterion(sr_value, target_value)

            loss = (l1 + l2) / 2
            if mean_loss is not None:
                mean_loss += loss
            else:
                mean_loss = loss

            with th.no_grad():
                target_value = target_aler(stimulus=stim.stimulus)[0]

        mean_loss /= len(groups)

        self.optimiser.zero_grad()
        mean_loss.backward()
        self.optimiser.step()

        print("Total loss = %.4e" % mean_loss.item())

        self.trajectory = []

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
        print("Max stimulus value: %.4f" % max_stim_value)
