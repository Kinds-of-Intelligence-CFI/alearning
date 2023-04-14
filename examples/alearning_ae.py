from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.alearner_ae import ALearnerAE
from animalai.envs.stimulus_ae import StimulusCategory
from animalai.envs.autoencoder import AutoEncoder
from gym_unity.envs import UnityToGymWrapper
import random
import os
import math
import time
import torch as th
import numpy as np
import argparse

PUNISHMENT = -10
N_EPISODES = 500
TOTAL_STIMULUS_CATEGORIES = 50

def run_agent_autoencoder(autoencoder_file: str,
                          n_channels, width, height,
                          configuration_file: str) -> None:
    """
    Loads a configuration file for a single arena and gives some example usage of mlagent python Low Level API
    See https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-API.md for details.
    For demo purposes uses a simple braitenberg vehicle-inspired agent that solves most tasks from category 1.
    """
    autoenc = AutoEncoder(n_channels, width, height).to(0)
    autoenc.load_state_dict(th.load(autoencoder_file))
    autoenc.eval()

    env_path = "../env/AnimalAI"

    configuration = configuration_file

    env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=configuration,
        seed=int(time.time()),
        play=False,
        useCamera=True,
        useRayCasts=False,
    )
    env = UnityToGymWrapper(env, uint8_visual=True, flatten_branched=True)
    print("Environment Loaded")

    alearner = ALearnerAE(env.action_space.n)

    mean = th.tensor([0.6282, 0.6240, 0.5943]).reshape(3, 1, 1).to(0)
    std = th.tensor([0.1751, 0.1605, 0.2117]).reshape(3, 1, 1).to(0)

    all_stimuli = []
    distance_criterion = True
    count_green = 0
    count_yellow = 0
    for _episode in range(N_EPISODES):
        # first observation
        res = env.step(0)
        obs = res[0]
        found_green = False
        found_yellow = False
        done = False
        reward = None
        episode_ended = False
        while not done:
            obs = np.expand_dims(obs, axis=0)
            obs = np.transpose(obs, axes=(0, 3, 1, 2))
            obs = (th.from_numpy(obs).to(0) - mean) / std
            candidate = autoenc(obs)[0].cpu().detach().numpy()
            added_to_category = False
            stimulus = None
            if not reward:
                if distance_criterion:
                    for stim in all_stimuli:
                        if stim.add_to_cluster(candidate):
                            added_to_category = True
                            stimulus = stim
                            break
                else:
                    min_idx = min(
                        map(lambda t: (t[0], t[1].distance(candidate)),
                            enumerate(all_stimuli)),
                        key=lambda t: t[1]
                    )[0]
                    stimulus = all_stimuli[min_idx]
                    stimulus.add_to_cluster(candidate)
                    added_to_category = True

            if not added_to_category:
                new_category = StimulusCategory(alearner, reward=reward)
                new_category.add_to_cluster(candidate)
                stimulus = new_category
                if reward is None:
                    all_stimuli.append(new_category)

            if len(all_stimuli) >= TOTAL_STIMULUS_CATEGORIES:
                for stim in all_stimuli:
                    stim.set_criterion()
                    distance_criterion = False

            if episode_ended:
                alearner.update_stimulus_values(stimulus)
                alearner.print_max_stim_val()
                done = True
            else:
                action = alearner.get_action(stimulus)
                res = env.step(action)
                obs = res[0]
                if not math.isclose(res[1], 0, abs_tol=1e-2):
                    reward = res[1]
                else:
                    reward = None
                episode_ended = res[2]
                if episode_ended:
                    if reward is None:
                        reward = PUNISHMENT
                    elif reward > 0:
                        found_green = True
                else:
                    if reward is not None and reward > 0:
                        found_yellow = True

        if found_green:
            print("found green")
            count_green += 1
        else:
            if found_yellow:
                print("found yellow")
                count_yellow += 1
            else:
                print("did not find reward")

        print("Stimuli count: %d" % len(all_stimuli))
        env.reset()
        alearner.decrease_temperature()

    env.close()
    print("Environment Closed")

    print("Success rate at finding green: %.4f" % (count_green / N_EPISODES))
    print("Success rate at finding yellow: %.4f" % (count_yellow / N_EPISODES))


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains alearning model using autoencoder for stimuli'
    )
    parser.add_argument('model_file', type=str, nargs='?',
                        default='autoencoder.pt')
    parser.add_argument('configuration_file', type=str, nargs='?')
    args = parser.parse_args()
    if args.configuration_file:
        configuration_file = args.configuration_file
    else:
        basic_tasks_folder = "../configs/basic/"
        configuration_files = os.listdir(basic_tasks_folder)
        configuration_random = random.randint(0, len(configuration_files))
        configuration_file = (
            basic_tasks_folder + configuration_files[configuration_random]
        )
    run_agent_autoencoder(autoencoder_file=args.model_file,
                          n_channels=3,
                          width=84,
                          height=84,
                          configuration_file=configuration_file)


# # # # # # # # # #
# # Observation examples
# obs = (env.get_steps(behavior)[0].obs)
# print(obs)
# o = env.getDict(obs)
# print(o["camera"])
# print(o["rays"])
# print("health: " + str(o["health"]))
# print("velocity: " + str(o["velocity"]))
# print("position: " + str(o["position"]))
# sys.exit()
# # # # # # # # # #
