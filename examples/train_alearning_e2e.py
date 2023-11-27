from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.alearner_e2e import ALearnerE2E
from animalai.envs.stimulus_e2e import StimulusDatapoint
from gym_unity.envs import UnityToGymWrapper
import math
import time
import os
import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

PUNISHMENT = -1
TRAINING_EPISODES = 40
TOTAL_EPISODES = 640
TOTAL_STAGE_3 = 600
CUT_OFF = 80
WINDOW_SIZE = 80
TRAIN_FREQUENCY = 5
DATASET_LIMIT = 10000
TRAJ_WINDOW_START = 5
MAX_TRAJ_WINDOW = 10


def save_first_frame(obs, episode):
    img = Image.fromarray(obs, 'RGB')
    img.save("debug/%d.png" % episode)


def normalise_obs(obs, mean, std, gpu=True):
    obs = np.expand_dims(obs, axis=0)
    obs = np.transpose(obs, axes=(0, 3, 1, 2))
    if gpu:
        obs = (th.from_numpy(obs).to(0) - mean) / std
    else:
        obs = (th.from_numpy(obs) - mean) / std

    return obs


def run_agent_e2e(n_channels, width, height,
                  config_file,
                  gpu=True) -> None:
    """
    Trains a-learning model with autoencoder on a set of training files
    and tests them on a set of test files
    """
    env_path = "../env/AnimalAI"

    mean = th.tensor([0.6282, 0.6240, 0.5943]).reshape(3, 1, 1)
    std = th.tensor([0.1751, 0.1605, 0.2117]).reshape(3, 1, 1)
    # mean = (0.6282, 0.6240, 0.5943)
    # std = (0.1751, 0.1605, 0.2117)
    # script = nn.Sequential(
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(degrees=(-180, 180)),
    #     transforms.Normalize(mean, std)
    # )
    if gpu:
        mean = mean.to(0)
        std = std.to(0)

    if not os.path.exists("debug"):
        os.makedirs("debug")

    env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=config_file,
        seed=int(time.time()),
        play=False,
        useCamera=True,
        useRayCasts=False,
        # inference=True
    )
    env = UnityToGymWrapper(env, uint8_visual=True, flatten_branched=True)
    print("Environment Loaded")

    alearner = ALearnerE2E(7, n_channels, width, height, gpu=gpu)

    total_green = 0
    total_episodes = 0

    runs_stage3 = []
    total_success_rate_stage3 = []
    window = []
    rolling_success_rate = []

    data = []
    # endpoints = []
    prev_stim = None
    prev_action = None
    old_obs = None
    old_reward = None
    # traj_window = TRAJ_WINDOW_START

    while total_episodes < TOTAL_EPISODES:
        res = env.step(0)
        obs = res[0]
        # save_first_frame(obs, total_episodes)
        normalised_obs = normalise_obs(obs, mean, std, gpu=gpu)
        episode_ended = False
        found_green = False
        done = False
        reward = None

        cand_data = []
        stimuli = set()

        if (
                (total_episodes <= TRAINING_EPISODES
                 and total_episodes % 10 == 0)
                or (total_episodes > TRAINING_EPISODES
                    and total_episodes < CUT_OFF
                    and (total_episodes - TRAINING_EPISODES) % 20 == 0)
        ):

        # this corresponds to an episode
        while not done:
            # obs = transforms.ToPILImage(mode="RGB")(obs)
            # obs = transforms.ToTensor()(obs)
            # obs = script(obs)
            # obs = obs[None, :]
            # if gpu:
            #     obs = obs.to(0)
            # onehot_stimulus = alearner.get_stimulus(obs)
            # stimulus = StimulusE2E(onehot_stimulus, reward=reward)

            if episode_ended:
                # alearner.update_stimulus_values(stimulus)
                done = True
                total_episodes += 1
            else:
                stim, action = alearner.get_action(normalised_obs,
                                                   reward=reward)
                stimuli.add(stim)
                res = env.step(action)
                obs = res[0]
                normalised_obs = normalise_obs(obs, mean, std, gpu=gpu)
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
                    print("Reward = %.4f" % reward)

                if prev_stim is not None:
                    d1 = StimulusDatapoint(img=old_obs, reward=old_reward)

                    if episode_ended:
                        d2 = StimulusDatapoint(reward=reward)
                    else:
                        d2 = StimulusDatapoint(img=obs, reward=reward)
                    cand_data.append((d1, prev_action, d2))
                    # cand_data.append((d1, prev_action, d2, 1))
                prev_stim = stim
                prev_action = action
                old_obs = obs
                old_reward = reward

        print("Number of different stimuli = %d" % len(stimuli))

        extended_cand_data = []
        for i, (d1, action, d2) in enumerate(cand_data):
            weight = 1 / (len(cand_data) - i)
            extended_cand_data.append((d1, action, d2, weight))
        data.extend(extended_cand_data)
        # data.extend(cand_data)

        if total_episodes > TRAINING_EPISODES:
            window.append(found_green)
            if found_green:
                total_green += 1

        print("Episode %d" % total_episodes)
        alearner.print_max_stim_val()

        env.reset()

        if total_episodes == TRAINING_EPISODES:
            # alearner.reset_optimiser()
            # traj_window = TRAJ_WINDOW_START
            # alearner.set_target_value()
            pass
        elif total_episodes > TRAINING_EPISODES:
            runs_stage3.append(total_episodes - TRAINING_EPISODES)
            success_rate = total_green / (total_episodes - TRAINING_EPISODES)
            total_success_rate_stage3.append(success_rate)
            print("Success rate = %.4f" % success_rate)

            window = window[-WINDOW_SIZE:]
            rolling_success_rate.append(sum(window) / len(window))

        if (
                total_episodes >= TRAINING_EPISODES
                and total_episodes % TRAIN_FREQUENCY == 0
        ):
            data = data[-DATASET_LIMIT:]
            alearner.do_training_round(data)
            alearner.do_training_round(data, l1_loss=False)

    print("Success rate at stage 3 = %.4f" % (total_green / TOTAL_STAGE_3))
    env.close()

    plt.plot(runs_stage3, total_success_rate_stage3)
    plt.title("Total success rate")
    plt.xlabel("total runs stage 3")
    plt.ylabel("success rate")
    plt.savefig("success_rate.png")

    plt.clf()
    plt.plot(runs_stage3, rolling_success_rate)
    plt.title("Rolling success rate (window size = 80)")
    plt.xlabel("total runs stage 3")
    plt.ylabel("rolling success rate")
    plt.savefig("rolling_success_rate.png")


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains alearning model using autoencoder for stimuli'
    )
    parser.add_argument('config_file', type=str, nargs='?',
                        default='../configs/mondragon/mondragon.yml')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.cpu:
        gpu = False
    else:
        gpu = True
    run_agent_e2e(n_channels=3,
                  width=84,
                  height=84,
                  config_file=args.config_file,
                  gpu=gpu)
