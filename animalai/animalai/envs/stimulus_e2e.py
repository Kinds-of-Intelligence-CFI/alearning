import torch as th

class StimulusE2E():
    def __init__(self, stimulus, reward=None):
        self.stimulus = stimulus
        max_idx = th.argmax(stimulus.detach().clone(), 1, keepdim=True)
        self.onehot = th.zeros(stimulus.shape)
        self.onehot.scatter_(1, max_idx, 1)
        self.onehot = self.onehot[0]
        # self.onehot = (stimulus.detach().clone() >= 0.5).int()[0]
        self.reward = reward

        # unconditioned value is assumed to be constant
        if self.reward is not None:
            self.u_val = self.reward
        else:
            self.u_val = 0

    def __hash__(self):
        to_hash = self.onehot.tolist()
        to_hash.append(self.u_val)
        to_hash = tuple(to_hash)
        return hash(to_hash)

    def __eq__(self, o):
        return self.__hash__() == o.__hash__()
