from animalai.envs.actions import AAIActions, AAIAction
from animalai.envs.raycastparser import RayCastParser
from animalai.envs.raycastparser import RayCastObjects
from collections import defaultdict
import numpy as np
from scipy.special import softmax
import random

class ALearner():
    """Implements the A-learning algorithm
    Can change the number of rays but only responds to GOODGOALs, GOODGOALMULTI and BADGOAL"""

    def __init__(self, no_rays, alpha_w=0.5,
                 alpha_v=0.5, beta=0.5, trace_size=5):
        self.no_rays = no_rays
        assert(self.no_rays % 2 == 1), "Only supports odd number of rays (but environment should only allow odd number"

        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.beta = beta

        self.w_values = {}
        self.sr_values = defaultdict(float)

        self.trace = []
        self.trace_size = trace_size
        self.trace_weights = np.arange(0, 1, 1 / self.trace_size)

        self.listOfObjects = [RayCastObjects.GOODGOAL, RayCastObjects.GOODGOALMULTI, RayCastObjects.BADGOAL, RayCastObjects.ARENA, RayCastObjects.IMMOVABLE, RayCastObjects.MOVABLE]
        self.raycast_parser = RayCastParser(self.listOfObjects, self.no_rays)
        self.actions = AAIActions()

        # list of stimulus-action pairs
        self.trajectory = []
        self.prev_stim = None

    def prettyPrint(self, obs) -> str:
        """Prints the parsed observation"""
        return self.raycast_parser.prettyPrint(obs)

    def get_action(self, stimulus) -> AAIAction:
        """Returns the action to take given the current parsed raycast observation"""
        self.prev_stim = stimulus
        self.trace.append(stimulus)
        self.trace = self.trace[-self.trace_size:]
        normalised_weights = softmax(self.trace_weights[-len(self.trace):])

        all_actions = self.actions.allActions
        all_keys = []
        for stim in self.trace:
            stim_keys = list(map(lambda a: (stim, a), all_actions))
            all_keys.append(stim_keys)

        all_sr_values = []
        for keys in all_keys:
            sr_values = np.fromiter(
                map(lambda k: self.sr_values[k], keys),
                dtype=float
            )
            all_sr_values.append(sr_values)

        average_sr_values = []

        probs = softmax(self.beta * all_sr_values)
        draw = random.random()
        idx = 0
        cum_prob = 0
        for prob in probs:
            cum_prob += prob
            if draw <= cum_prob:
                break
            idx += 1

        action = all_actions[idx]
        self.trajectory.append((self.prev_stim, action))
        return action

    def ahead(self, obs, obj):
        """Returns true if the input object is ahead of the agent"""
        if(obs[self.listOfObjects.index(obj)][int((self.no_rays-1)/2)] > 0):
            # print("found " + str(object) + " ahead")
            return True
        return False

    def left(self, obs, obj):
        """Returns true if the input object is to the left of the agent"""
        for i in range(int((self.no_rays-1)/2)):
            if(obs[self.listOfObjects.index(obj)][i] > 0):
                return True
        return False

    def right(self, obs, obj):
        """Returns true if the input object to the right of the agent"""
        for i in range(int((self.no_rays-1)/2)):
            if(obs[self.listOfObjects.index(obj)][i+int((self.no_rays-1)/2) + 1] > 0):
                return True
        return False

    def update_stimulus_values(self, final_stim):
        self.trajectory.reverse()

        if final_stim not in self.w_values and final_stim is not None:
            self.w_values[final_stim] = final_stim.u_val
        next_stim = final_stim
        for stim, action in self.trajectory:
            if stim is not None:
                if stim not in self.w_values:
                    self.w_values[stim] = stim.u_val
                self.w_values[stim] += self.alpha_w * \
                    (next_stim.u_val + self.w_values[next_stim] -
                    self.w_values[stim])

                self.sr_values[(stim, action)] += self.alpha_v * \
                    (next_stim.u_val + self.w_values[next_stim] -
                    self.sr_values[(stim, action)])

                next_stim = stim
        print("Stimuli count: %d" % len(self.w_values.keys()))
        self.trajectory = []

    def increase_exploit(self):
        self.beta += 1

    def double_exploit(self):
        self.beta *= 2

    def print_maps(self, all_maps=False):
        max_stim = max(self.w_values, key=lambda k: self.w_values[k])
        print("Stimulus value: %.4f" % self.w_values[max_stim])
        print(max_stim)

        # for stim, action in self.sr_values:
        #     print("S-R value: %.4f" % self.sr_values[(stim, action)])
        #     print("Action: ", action)
        #     print(stim)
