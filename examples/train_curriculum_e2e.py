from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.alearner_e2e import ALearnerE2E
from animalai.envs.stimulus_e2e import StimulusDatapoint
from gym_unity.envs import UnityToGymWrapper
import gym
import random
import math
import time
import os
import torch as th
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

PUNISHMENT = -1
WINDOW_SIZE = 80
TRAIN_FREQUENCY = 10
# TRAIN_LIMIT = 50
DATASET_LIMIT = 10000
NUM_FRAMES = 4
N_DATAPOINTS = 50

N_TASKS = {
    "task_type_1_2": 216,
    "task_type_3_4": 993,
    "task_type_5_6": 1030,
    "task_type_7_8": 993
}
REPS = {
    "task_type_1_2": 5,
    "task_type_3_4": 1,
    "task_type_5_6": 1,
    "task_type_7_8": 1
}
TASK_FILE = "all_tasks.yml"

PROP_OLD_DATA = 0.2
SAVE_PREV_DATA = 0.1

# def save_first_frame(obs, episode):
#     img = Image.fromarray(obs, 'RGB')
#     img.save("debug/%d.png" % episode)

def transpose_vector_obs(obs):
    # obs = np.expand_dims(obs, axis=0)
    # obs = np.transpose(obs, axes=(0, 4, 1, 2, 3))
    obs = np.transpose(obs, axes=(3, 0, 1, 2))

    return obs


def normalise_obs(obs, mean, std, gpu=True):
    obs = np.expand_dims(obs, axis=0)
    if gpu:
        obs = (th.from_numpy(obs).to(0) - mean) / std
    else:
        obs = (th.from_numpy(obs) - mean) / std

    return obs


def run_agent_e2e(n_channels, width, height,
                  curriculum_dir,
                  model_file=None,
                  gpu=True) -> None:
    """
    Trains a-learning model with autoencoder on a set of training files
    and tests them on a set of test files
    """
    env_path = "../env/AnimalAI"

    mean = th.tensor([0.6282, 0.6240, 0.5943]).reshape(3, 1, 1, 1)
    std = th.tensor([0.1751, 0.1605, 0.2117]).reshape(3, 1, 1, 1)
    if gpu:
        mean = mean.to(0)
        std = std.to(0)

    if not os.path.exists("e2e_plots"):
        os.makedirs("e2e_plots")

    alearner = ALearnerE2E(7, n_channels, width, height,
                           gpu=gpu, model_file=model_file)

    components = curriculum_dir.split("/")
    if components[-1] == "":
        task = components[-2]
    else:
        task = components[-1]

    config_file = os.path.join(curriculum_dir, TASK_FILE)
    env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=config_file,
        seed=int(time.time()),
        play=False,
        useCamera=True,
        useRayCasts=False,
        base_port=6000+random.randint(1, 1000),
        # inference=True
    )
    env = UnityToGymWrapper(env, uint8_visual=True, flatten_branched=True)
    env = gym.wrappers.FrameStack(env, NUM_FRAMES)
    print("Environment Loaded")
    total_green = 0
    total_episodes = 0

    runs = []
    total_success_rate = []
    window = []
    rolling_success_rate = []

    meta_data = None
    with open(
            os.path.join(curriculum_dir, "meta_data.bin"), "rb"
    ) as fin:
        meta_data = pickle.load(fin)

    log_file = open(
        os.path.join(curriculum_dir, 'results_e2e.csv'), "w"
    )

    data = []
    prev_stim = None
    prev_action = None
    old_obs = None
    old_reward = None

    use_estimates = True

    n_reps = REPS[task]
    for k in range(n_reps):

        if k > 1:
            use_estimates = True
        else:
            alearner.reset_temperature()

        n_episodes = 0
        while n_episodes < N_TASKS[task]:
            obs = env.reset()
            obs = transpose_vector_obs(obs)
            normalised_obs = normalise_obs(obs, mean, std, gpu=gpu)
            episode_ended = False
            found_green = False
            done = False
            reward = None

            cand_data = []
            stimuli = set()

            # this corresponds to an episode
            last_point = None
            while not done:
                if episode_ended:
                    done = True
                    n_episodes += 1
                    total_episodes += 1
                else:
                    stim, action = alearner.get_action(normalised_obs,
                                                       reward=reward)
                    stimuli.add(stim)
                    res = env.step(action)
                    obs = res[0]
                    obs = transpose_vector_obs(obs)
                    normalised_obs = normalise_obs(obs, mean, std, gpu=gpu)
                    if not math.isclose(res[1], 0, abs_tol=1e-2):
                        reward = res[1]
                    else:
                        reward = None
                    episode_ended = res[2]
                    if episode_ended:
                        if reward is not None and reward > 0:
                            found_green = True
                            reward = 1
                            print("found green")

                    if prev_stim is not None:
                        d1 = StimulusDatapoint(img=old_obs,
                                               reward=old_reward)

                        if episode_ended and reward is not None:
                            d2 = StimulusDatapoint(reward=reward)
                        else:
                            d2 = StimulusDatapoint(img=obs, reward=reward)
                        cand_data.append((d1, prev_action, d2))
                        last_point = d2
                        # cand_data.append((d1, prev_action, d2, 1))
                    prev_stim = stim
                    prev_action = action
                    old_obs = obs
                    old_reward = reward

            print("Number of different stimuli = %d" % len(stimuli))

            if episode_ended and (reward is not None or use_estimates):
                extended_cand_data = []
                # d1 = cand_data[0][0]
                # i = 5
                # while i < len(cand_data):
                #     if i == len(cand_data) - 1:
                #         last_point = None
                #     action = cand_data[i][1]
                #     d2 = cand_data[i][2]
                #     extended_cand_data.append((d1, action, d2, 1, last_point))
                #     d1 = d2
                #     i += 5

                for i, (d1, action, d2) in enumerate(cand_data):
                    weight = 1
                    if i == len(cand_data) - 1:
                        last_point = None
                    extended_cand_data.append((d1, action, d2,
                                               weight, last_point))
                data.extend(extended_cand_data)

            window.append(found_green)
            if found_green:
                total_green += 1
                alearner.exploit()
            alearner.decrease_temperature()
            if k == n_reps - 1:
                line = ("%d," % found_green) + ",".join(
                    meta_data[n_episodes - 1]
                ) + "/n"
                log_file.write(line)

            print("Episode %d" % total_episodes)
            alearner.print_max_stim_val()

            env.reset()

            runs.append(total_episodes)
            success_rate = total_green / total_episodes
            total_success_rate.append(success_rate)
            print("Success rate = %.4f" % success_rate)

            window = window[-WINDOW_SIZE:]
            rolling_success_rate.append(sum(window) / len(window))

            if total_episodes % TRAIN_FREQUENCY == 0:
                if len(data) > DATASET_LIMIT:
                    train_data = random.sample(data, DATASET_LIMIT)
                else:
                    train_data = data[:]
                # if prev_data:
                #     n_select = int(DATASET_LIMIT * PROP_OLD_DATA)
                #     if len(prev_data) >= n_select:
                #         random_selection = random.sample(prev_data,
                #                                          n_select)
                #         train_data.extend(random_selection)
                #     else:
                #         train_data.extend(prev_data)
                alearner.do_training_round(train_data)
                # alearner.do_training_round(train_data, l1_loss=False)

    print("Success rate = %.4f" % (total_green / total_episodes))
    env.close()

    plt.plot(runs, total_success_rate)
    plt.title("Total success rate")
    plt.xlabel("total runs")
    plt.ylabel("success rate")
    plt.savefig(("e2e_plots/%s_success_rate.png") % task)

    plt.clf()
    plt.plot(runs, rolling_success_rate)
    plt.title("Rolling success rate (window size = 80)")
    plt.xlabel("total runs")
    plt.ylabel("rolling success rate")
    plt.savefig(("e2e_plots/%s_rolling_success_rate.png") % task)

    # n_select = int(SAVE_PREV_DATA * len(data))
    # random_selection = random.sample(data, n_select)
    # prev_data.extend(random_selection)
    log_file.close()

    alearner.save_model()


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains e2e alearning model on basic curriculum'
    )
    parser.add_argument('curriculum_dir', type=str)
    parser.add_argument('model_file', type=str, nargs='?', default='e2e.pt')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.cpu:
        gpu = False
    else:
        gpu = True
    run_agent_e2e(n_channels=3,
                  width=84,
                  height=84,
                  curriculum_dir=args.curriculum_dir,
                  model_file=args.model_file,
                  gpu=gpu)
