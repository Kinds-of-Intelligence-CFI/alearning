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
from PIL import Image

PUNISHMENT = -1
TRAINING_EPISODES = 40
TOTAL_EPISODES = 640
TOTAL_STAGE_3 = 600
CUT_OFF = 120
PARTIAL_CATEGORIES = 10
CHECKPOINT_CATEGORIES = 20
TOTAL_CATEGORIES = 30


def save_first_frame(obs, episode):
    img = Image.fromarray(obs, 'RGB')
    img.save("debug/%d.png" % episode)


def run_agent_autoencoder(autoencoder_file: str,
                          n_channels, width, height,
                          config_file,
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

    if alearner is None:
        # alearner = ALearnerAE(env.action_space.n)
        alearner = ALearnerAE(7)

    total_green = 0
    total_episodes = 0

    runs_stage3 = []
    rolling_success_rate_stage3 = []

    total_categories = PARTIAL_CATEGORIES
    while total_episodes < TOTAL_EPISODES:
        res = env.step(0)
        obs = res[0]
        save_first_frame(obs, total_episodes)
        episode_ended = False
        found_green = False
        done = False
        reward = None

        if (
                (total_episodes <= TRAINING_EPISODES
                 and total_episodes % 10 == 0)
                or (total_episodes > TRAINING_EPISODES
                    and total_episodes < CUT_OFF
                    and (total_episodes - TRAINING_EPISODES) % 20 == 0)
        ):
            alearner.reset_temperature()

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

        if found_green:
            if total_episodes > TRAINING_EPISODES:
                total_green += 1
        alearner.decrease_temperature()

        print("Episode %d | stimuli count: %d"
              % (total_episodes, len(all_stimuli)))
        alearner.print_max_stim_val()

        if total_episodes == TRAINING_EPISODES:
            for stim in all_stimuli:
                stim.set_criterion(True)
            distance_criterion = True
            total_categories = CHECKPOINT_CATEGORIES
        elif total_episodes == TRAINING_EPISODES + 20:
            for stim in all_stimuli:
                stim.set_criterion(True)
            distance_criterion = True
            total_categories = TOTAL_CATEGORIES

        env.reset()

        if total_episodes > TRAINING_EPISODES:
            runs_stage3.append(total_episodes - TRAINING_EPISODES)
            success_rate = total_green / (total_episodes - TRAINING_EPISODES)
            rolling_success_rate_stage3.append(success_rate)
            print("Success rate = %.4f" % success_rate)

    print("Success rate at stage 3 = %.4f" % (total_green / TOTAL_STAGE_3))
    env.close()

    plt.plot(runs_stage3, rolling_success_rate_stage3)
    plt.xlabel("total runs stage 3")
    plt.ylabel("success rate")
    plt.savefig("success_rate.png")


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains alearning model using autoencoder for stimuli'
    )
    parser.add_argument('config_file', type=str, nargs='?',
                        default='../configs/mondragon/mondragon.yml')
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
                          config_file=args.config_file,
                          gpu=gpu)

