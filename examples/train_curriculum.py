from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.alearner_ae import ALearnerAE
from animalai.envs.stimulus_ae import StimulusCategory
from animalai.envs.autoencoder import AutoEncoder
from gym_unity.envs import UnityToGymWrapper
import math
import time
import os
import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle

PUNISHMENT = -1
START_CATEGORIES = 10
WINDOW_SIZE = 80
N_TASKS = {
    "task_type_1_2": 216,
    "task_type_3_GR_Platform_2x9_HC": 324,
    "task_type_3_GR_Platform_9x9_HC": 108,
    "task_type_3_GR_Platform_4x4_HC": 972,
    "task_type_4_GR_Platform_20x20_FA": 432,
    "task_type_4_GR_Platform_9x9_FA": 432,
    "task_type_4_GR_Platform_3x40_FA": 216,
    "task_type_5_6": 1030,
    "task_type_7_8": 993
}
REPS = {
    "task_type_1_2": 4,
    "task_type_3_GR_Platform_2x9_HC": 1,
    "task_type_3_GR_Platform_9x9_HC": 1,
    "task_type_3_GR_Platform_4x4_HC": 1,
    "task_type_4_GR_Platform_20x20_FA": 1,
    "task_type_4_GR_Platform_9x9_FA": 1,
    "task_type_4_GR_Platform_3x40_FA": 1,
    "task_type_5_6": 1,
    "task_type_7_8": 1
}
TASK_FILE = "all_tasks.yml"


def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, "rb") as fin:
            state = pickle.load(fin)
        total_categories = state["total_categories"]
        all_stimuli = state["all_stimuli"]
        alearner = state["alearner"]
    else:
        total_categories = START_CATEGORIES
        all_stimuli = []
        alearner = ALearnerAE(7)

    return total_categories, all_stimuli, alearner


def save_state(state_file, total_categories, all_stimuli, alearner):
    state = {
        "total_categories": total_categories,
        "all_stimuli": all_stimuli,
        "alearner": alearner
    }

    with open(state_file, "wb") as fout:
        pickle.dump(state, fout)


def run_agent_autoencoder(autoencoder_file: str,
                          n_channels, width, height,
                          curriculum_dir,
                          state_file,
                          gpu=True) -> None:
    """
    Trains a-learning model with autoencoder on a set of training files
    and tests them on a set of test files
    """
    autoenc = AutoEncoder(n_channels, width, height)
    if gpu:
        autoenc = autoenc.to(0)
        autoenc.load_state_dict(th.load(autoencoder_file))
    else:
        autoenc.load_state_dict(th.load(autoencoder_file,
                                        map_location=th.device('cpu')))
    autoenc.eval()

    env_path = "../env/AnimalAI"

    mean = th.tensor([0.6282, 0.6240, 0.5943]).reshape(3, 1, 1)
    std = th.tensor([0.1751, 0.1605, 0.2117]).reshape(3, 1, 1)
    if gpu:
        mean = mean.to(0)
        std = std.to(0)

    if not os.path.exists("plots"):
        os.makedirs("plots")

    total_categories, all_stimuli, alearner = load_state(state_file)

    components = curriculum_dir.split("/")
    if components[-1] == "":
        task = components[-2]
    else:
        task = components[-1]

    distance_criterion = True
    config_file = os.path.join(curriculum_dir, TASK_FILE)
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

    log_file = open(os.path.join(curriculum_dir, 'results.csv'), "w")

    n_reps = REPS[task]

    for k in range(n_reps):
        alearner.reset_temperature()

        n_episodes = 0
        while n_episodes < N_TASKS[task]:
            res = env.step(0)
            obs = res[0]
            # save_first_frame(obs, total_episodes)
            episode_ended = False
            found_green = False
            done = False
            reward = None

            # this corresponds to an episode
            while not done:
                obs = np.expand_dims(obs, axis=0)
                obs = np.transpose(obs, axes=(0, 3, 1, 2))
                if gpu:
                    obs = (th.from_numpy(obs).to(0) - mean) / std
                else:
                    obs = (th.from_numpy(obs) - mean) / std
                candidate = autoenc(obs)[0].cpu().detach().numpy()
                added_to_category = False
                stimulus = None
                if not reward:
                    if all_stimuli:
                        min_idx = min(
                            map(lambda t: (t[0], t[1].distance(candidate)),
                                enumerate(all_stimuli)),
                            key=lambda t: t[1]
                        )[0]
                    else:
                        min_idx = -1
                    if distance_criterion:
                        if (
                                min_idx > 0 and
                                all_stimuli[min_idx].add_to_cluster(candidate)
                        ):
                            added_to_category = True
                            stimulus = all_stimuli[min_idx]
                    else:
                        stimulus = all_stimuli[min_idx]
                        stimulus.add_to_cluster(candidate)
                        added_to_category = True

                if not added_to_category:
                    new_category = StimulusCategory(
                        alearner, reward=reward,
                        distance_criterion=distance_criterion
                    )
                    new_category.add_to_cluster(candidate)
                    stimulus = new_category
                    if reward is None:
                        all_stimuli.append(new_category)

                if len(all_stimuli) >= total_categories:
                    for stim in all_stimuli:
                        stim.set_criterion(False)
                    distance_criterion = False

                if episode_ended:
                    alearner.update_stimulus_values(stimulus)
                    done = True
                    n_episodes += 1
                    total_episodes += 1
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
                        if reward is not None and reward > 0:
                            found_green = True

            window.append(found_green)
            if found_green:
                total_green += 1
            alearner.decrease_temperature()
            if k == n_reps - 1:
                line = ("%d," % found_green) + ",".join(
                    meta_data[n_episodes - 1]
                ) + "\n"
                log_file.write(line)

            print("Episode %d | stimuli count: %d"
                    % (n_episodes, len(all_stimuli)))
            alearner.print_max_stim_val()

            env.reset()

            runs.append(total_episodes)
            success_rate = total_green / total_episodes
            total_success_rate.append(success_rate)
            print("Success rate = %.4f" % success_rate)

            window = window[-WINDOW_SIZE:]
            rolling_success_rate.append(sum(window) / len(window))

    for stim in all_stimuli:
        stim.set_criterion(True)
        distance_criterion = True
    total_categories += 10

    print("Final success rate = %.4f" % (total_green / total_episodes))
    env.close()

    plt.clf()
    plt.plot(runs, total_success_rate)
    plt.title("Total success rate")
    plt.xlabel("total runs")
    plt.ylabel("success rate")
    plt.savefig(("plots/%s_success_rate.png") % task)

    plt.clf()
    plt.plot(runs, rolling_success_rate)
    plt.title("Rolling success rate (window size = 80)")
    plt.xlabel("total runs")
    plt.ylabel("rolling success rate")
    plt.savefig(("plots/%s_rolling_success_rate.png") % task)

    log_file.close()
    save_state(state_file, total_categories, all_stimuli, alearner)


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains alearning model using autoencoder for stimuli'
    )
    parser.add_argument('curriculum_dir', type=str)
    parser.add_argument('model_file', type=str, nargs='?',
                        default='autoencoder.pt')
    parser.add_argument('state_file', type=str, nargs='?',
                        default='state.bin')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.cpu:
        gpu = False
    else:
        gpu = True
    run_agent_autoencoder(autoencoder_file=args.model_file,
                          n_channels=3,
                          width=84,
                          height=84,
                          curriculum_dir=args.curriculum_dir,
                          state_file=args.state_file,
                          gpu=gpu)
