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
from PIL import Image

PUNISHMENT = -1
WINDOW_SIZE = 80
TRAIN_LIMIT = 100
DATASET_LIMIT = 10000
NUM_FRAMES = 4
N_DATAPOINTS = 50
LENGTH_ACTION = 10

REPS = {
    "base_training": 50,
    "task_type_1_2": 2,
    "task_type_1_2_test": 2,
    "task_type_3_4": 1,
    "task_type_3_4_test": 1,
    "task_type_5_6": 1,
    "task_type_5_6_test": 1,
    "task_type_7_8": 1
}
TASK_FILE = "all_tasks.yml"

PROP_OLD_DATA = 0.2
SAVE_PREV_DATA = 0.1


def save_first_frame(obs, episode):
    if not os.path.exists("debug"):
        os.makedirs("debug")
    img = Image.fromarray(obs, 'RGB')
    img.save("debug/%d.png" % episode)


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
                  test_freq=10,
                  gpu=True,
                  test=False,
                  future=False) -> None:
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

    base_dir = "all_results/test_freq=%d/future_discount=%d" \
        % (test_freq, future)
    if not os.path.exists(os.path.join(base_dir, "e2e_plots")):
        os.makedirs(os.path.join(base_dir, "e2e_plots"))

    if not os.path.exists(os.path.join(base_dir, "results_e2e")):
        os.makedirs(os.path.join(base_dir, "results_e2e"))

    alearner = ALearnerE2E(7, n_channels, width, height,
                           gpu=gpu, future=future,
                           model_file=os.path.join(base_dir, model_file))

    components = curriculum_dir.split("/")
    if components[-1] == "":
        task = components[-3] if not test else components[-2]
    else:
        task = components[-2] if not test else components[-1]

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

    meta_data = None
    with open(
            os.path.join(curriculum_dir, "meta_data.bin"), "rb"
    ) as fin:
        meta_data = pickle.load(fin)

    train_tasks = None
    test_tasks = None
    if test:
        total_tasks = meta_data[0]
        meta_data = meta_data[1:]
    else:
        train_tasks = meta_data[0]
        test_tasks = meta_data[1]
        total_tasks = meta_data[2]
        meta_data = meta_data[3:]

    if test:
        log_file = open(
            os.path.join(base_dir, "results_e2e", "%s_testset.csv" % task), "w"
        )
    else:
        log_file = open(
            os.path.join(base_dir, "results_e2e", "%s_results.csv" % task), "w"
        )
    log_file.write("found_green,task_type,reward_size," +
                   "left_right_of_agent,distance_from_agent_x," +
                   "position_to_agent,distance_from_agent_z,platform_wall," +
                   "x_z_wall_dimensions,wall_location_x,wall_location_z," +
                   "platform_wall,x_z_wall_dimensions,wall_location_x," +
                   "wall_location_z,platform_wall,x_z_wall_dimensions," +
                   "wall_location_x,wall_location_z," +
                   "agent_floor_plat,goal_floor_plat\n")

    data = []
    prev_stim = None
    prev_action = None
    old_obs = None
    old_reward = None

    use_estimates = True
    test_runs = []
    test_hit_rate = []
    n_test_runs = 0

    n_reps = REPS[task]
    for k in range(n_reps):

        n_episodes = 0
        n_train_episodes = 0
        n_test_episodes = 0
        test_green = 0
        testing = False
        while n_episodes < total_tasks:
            if not test:
                if testing:
                    if n_test_episodes % test_tasks == 0:
                        testing = False
                        n_test_runs += 1
                        hit_rate = test_green / test_tasks
                        print("Test %d: hit rate = %.4f"
                              % (n_test_runs, hit_rate))
                        test_runs.append(n_test_runs)
                        test_hit_rate.append(hit_rate)
                elif ((n_episodes % (test_tasks + test_freq) == 0)
                      or n_train_episodes >= train_tasks):
                    testing = True
                    n_test_episodes = 0
                    test_green = 0
            obs = env.reset()
            save_first_frame(obs[0], total_episodes)
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
                    if testing:
                        n_test_episodes += 1
                    else:
                        n_train_episodes += 1
                else:
                    stim, action = alearner.get_action(normalised_obs,
                                                       reward=reward)
                    stimuli.add(stim)
                    for _ in range(LENGTH_ACTION):
                        res = env.step(action)
                        if res[2]:
                            break
                    obs = res[0]
                    obs = transpose_vector_obs(obs)
                    normalised_obs = normalise_obs(obs, mean, std, gpu=gpu)
                    if not math.isclose(res[1], 0, abs_tol=1e-2):
                        reward = res[1]
                    else:
                        reward = 0
                    episode_ended = res[2]
                    if episode_ended:
                        if reward is not None and reward > 0:
                            found_green = True
                            print("found green | reward = %.2f" % reward)

                    if not testing and not test:
                        if prev_stim is not None:
                            d1 = StimulusDatapoint(img=old_obs,
                                                   reward=old_reward)

                            if episode_ended and reward is not None:
                                d2 = StimulusDatapoint(reward=reward)
                            else:
                                d2 = StimulusDatapoint(img=obs, reward=reward)
                            cand_data.append((d1, prev_action, d2))
                            last_point = d2
                        prev_stim = stim
                        prev_action = action
                        old_obs = obs
                        old_reward = reward

            print("Number of different stimuli = %d" % len(stimuli))

            if not testing and not test:
                if episode_ended and (reward is not None or use_estimates):
                    extended_cand_data = []

                    for i, (d1, action, d2) in enumerate(cand_data):
                        weight = 1 / (len(cand_data) - i)
                        if i == len(cand_data) - 1:
                            last_point = None
                        extended_cand_data.append((d1, action, d2,
                                                   weight, last_point))
                    data.extend(extended_cand_data)

            if found_green:
                total_green += 1
                if testing:
                    test_green += 1

            if testing or test:
                line = ("%d," % found_green) + ",".join(
                    meta_data[n_episodes - 1]
                ) + "\n"
                log_file.write(line)

            print("Episode %d" % total_episodes)
            alearner.print_max_stim_val()

            env.reset()

            if not testing and not test and (
                    n_train_episodes % test_freq == 0
                    or n_train_episodes >= train_tasks
            ):
                if len(data) > DATASET_LIMIT:
                    train_data = random.sample(data, DATASET_LIMIT)
                else:
                    train_data = data[:]
                alearner.do_training_round(train_data)

    print("Success rate = %.4f" % (total_green / total_episodes))
    env.close()

    if testing:
        plt.clf()
        plt.scatter(test_runs, test_hit_rate)
        plt.title("Test set success rate")
        plt.xlabel("test run")
        plt.ylabel("success rate")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(base_dir, "e2e_plots",
                                 "%s_test_hit_rate.png" % task))

    log_file.close()

    alearner.save_model()


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains e2e alearning model on basic curriculum'
    )
    parser.add_argument('curriculum_dir', type=str)
    parser.add_argument('test_freq', type=int, nargs='?', default=10)
    parser.add_argument('model_file', type=str, nargs='?', default='e2e.pt')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--future', action='store_true')
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
                  test_freq=args.test_freq,
                  gpu=gpu,
                  test=args.test,
                  future=args.future)
