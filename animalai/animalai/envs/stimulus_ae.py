from scipy.spatial.distance import cosine
import numpy as np
import uuid

CRITERION = 0.05


class StimulusCategory():
    def __init__(self, agent, reward=None, distance_criterion=True):
        self.id = uuid.uuid4()
        self.agent = agent
        self.encoded_centroid = None
        self.n_members = 0
        self.reward = reward
        self.distance_criterion = distance_criterion

        # unconditioned value is assumed to be constant
        if self.reward is not None:
            self.u_val = self.reward
        else:
            self.u_val = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, o):
        return self.id == o.id

    def distance(self, encoded):
        norm_centroid = self.encoded_centroid \
            / np.linalg.norm(self.encoded_centroid)
        norm_encoded = encoded / np.linalg.norm(encoded)
        return cosine(norm_centroid, norm_encoded)

    def set_criterion(self, dist=True):
        self.distance_criterion = dist

    def add_to_cluster(self, encoded):
        if self.encoded_centroid is not None:
            if (
                    not self.distance_criterion
                    or self.distance(encoded) <= CRITERION
            ):
                self.encoded_centroid = \
                    (self.n_members * self.encoded_centroid + encoded) \
                    / (self.n_members + 1)
                self.n_members += 1
                return True
            else:
                return False
        else:
            self.encoded_centroid = encoded
            self.n_members = 1
            return True

