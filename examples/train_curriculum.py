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
# SUB_DIRS = ["task_type_1_GR_HC", "task_type_2_GR_FA",
#             "task_type_3_GR_Platform_HC", "task_type_4_GR_Platform_FA",
#             "task_type_5_GR_Wall_HC", "task_type_6_GR_Wall_FA",
#             "task_type_7_GR_Platform_Wall_HC",
#             "task_type_8_GR_Platform_Wall_FA"]
SUB_DIRS = ["task_type_1_2", "task_type_3_4"]
# N_TASKS = [108, 108, 1404, 1080, 2808, 2916, 32292, 57996]
N_TASKS = [216, 2484]
REPS = [4, 1]
TASK_FILE = "all_tasks.yml"


# def save_first_frame(obs, episode):
#     img = Image.fromarray(obs, 'RGB')
#     img.save("debug/%d.png" % episode)


def run_agent_autoencoder(autoencoder_file: str,
                          n_channels, width, height,
                          curriculum_dir,
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

    alearner = None

    all_stimuli = []
    distance_criterion = True

    if not os.path.exists("plots"):
        os.makedirs("plots")

    alearner = ALearnerAE(7)

    total_categories = START_CATEGORIES
    for i, task in enumerate(SUB_DIRS):
        config_file = os.path.join(curriculum_dir, task, "all_tasks.yml")
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
                os.path.join(curriculum_dir, task, "meta_data.bin"), "rb"
        ) as fin:
            meta_data = pickle.load(fin)

        log_file = open(os.path.join(curriculum_dir, task, 'results.csv'), "w")

        n_reps = REPS[i]

        for k in range(n_reps):
            if i < 4:
                alearner.reset_temperature()

            n_episodes = 0
            while n_episodes < N_TASKS[i]:
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
                            if reward is None:
                                reward = PUNISHMENT
                            elif reward > 0:
                                found_green = True

                window.append(found_green)
                if found_green:
                    total_green += 1
                alearner.decrease_temperature()
                if k == n_reps - 1:
                    line = ",".join(meta_data[n_episodes - 1]) \
                        + (",%d\n" % found_green)
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


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains alearning model using autoencoder for stimuli'
    )
    parser.add_argument('curriculum_dir', type=str, nargs='?',
                        default='../configs/basic_curriculum')
    parser.add_argument('model_file', type=str, nargs='?',
                        default='autoencoder.pt')
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
                          gpu=gpu)
