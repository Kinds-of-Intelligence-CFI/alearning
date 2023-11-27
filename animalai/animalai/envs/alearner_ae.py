from collections import defaultdict
import numpy as np
import random
from itertools import groupby


class ALearnerAE():
    """Implements the A-learning algorithm
    Can change the number of rays but only responds to GOODGOALs, GOODGOALMULTI and BADGOAL"""

    def __init__(self, n_actions, alpha_w=0.5, alpha_v=0.5, epsilon=0.8):
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.epsilon = epsilon

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
        all_keys = list(map(lambda a: (self.prev_stim.id, a),
                            range(self.n_actions)))

        all_sr_values = np.fromiter(
            map(lambda k: self.sr_values[k], all_keys),
            dtype=float
        )

        draw = random.random()
        if draw <= self.epsilon:
            max_idx = np.argmax(all_sr_values)
            action = all_keys[max_idx][1]
        else:
            action = random.randrange(0, self.n_actions)
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
            self.w_values[stim.id] += self.alpha_w * \
                (next_stim.u_val + self.w_values[next_stim.id] -
                 self.w_values[stim.id])

            stim, action = grouped_pairs[i][0]
            self.sr_values[(stim.id, action)] += self.alpha_v * \
                (next_stim.u_val + self.w_values[next_stim.id] -
                    self.sr_values[(stim.id, action)])

            next_stim = stim
        self.trajectory = []

    def print_max_stim_val(self):
        if self.w_values:
            max_stim_value = max(self.w_values.values())
            print("Max stimulus value: %.4f" % max_stim_value)
