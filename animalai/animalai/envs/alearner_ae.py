from collections import defaultdict
import numpy as np
from scipy.special import softmax
import random
from itertools import groupby


class ALearnerAE():
    """Implements the A-learning algorithm
    Can change the number of rays but only responds to GOODGOALs, GOODGOALMULTI and BADGOAL"""

    def __init__(self, n_actions, alpha_w=0.9, alpha_v=0.9, temperature=100):
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.temperature = temperature
        self.initial_temperature = temperature

        self.w_values = defaultdict(float)
        self.sr_values = defaultdict(float)

        self.n_actions = n_actions

        # list of stimulus-action pairs
        self.trajectory = []
        self.prev_stim = None

    def prettyPrint(self, obs) -> str:
        """Prints the parsed observation"""
        return self.raycast_parser.prettyPrint(obs)

    def get_action(self, stimulus) -> int:
        """Returns the action to take given the current parsed raycast observation"""
        self.prev_stim = stimulus
        all_keys = list(map(lambda a: (self.prev_stim, a),
                            range(self.n_actions)))

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

        self.trajectory.append((self.prev_stim, action))
        return action

    def update_stimulus_values(self, final_stim):
        grouped_stimuli = []
        grouped_pairs = []

        for k, group in groupby(self.trajectory, key=lambda x: x[0]):
            grouped_stimuli.append(k)
            group = list(group)
            group.reverse()
            grouped_pairs.append(group)

        grouped_stimuli.reverse()
        grouped_pairs.reverse()

        next_stim = final_stim
        for i, stim in enumerate(grouped_stimuli):
            self.w_values[stim] += self.alpha_w * \
                (next_stim.u_val + self.w_values[next_stim] -
                 self.w_values[stim])

            stim, action = grouped_pairs[i][0]
            self.sr_values[(stim, action)] += self.alpha_v * \
                (next_stim.u_val + self.w_values[next_stim] -
                    self.sr_values[(stim, action)])

            next_stim = stim
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
