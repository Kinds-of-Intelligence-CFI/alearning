from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.alearner import ALearner
from animalai.envs.stimulus import Stimulus
import sys
import random
import os
import math
import time

PUNISHMENT = -2

def run_agent_single_config(configuration_file: str) -> None:
    """
    Loads a configuration file for a single arena and gives some example usage of mlagent python Low Level API
    See https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-API.md for details.
    For demo purposes uses a simple braitenberg vehicle-inspired agent that solves most tasks from category 1.
    """
    env_path = "../env/AnimalAI"

    configuration = configuration_file

    totalRays = 9
    env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=configuration,
        seed=int(time.time()),
        play=False,
        useCamera=False, #The Braitenberg agent works with raycasts
        useRayCasts=True,
        raysPerSide=int((totalRays-1)/2),
        rayMaxDegrees = 30,
    )
    print("Environment Loaded")

    alearner = ALearner(totalRays) #A simple BraitenBerg Agent that heads towards food items.
    behavior = list(env.behavior_specs.keys())[0] # by default should be AnimalAI?team=0

    with open("stimuli.log", "w") as fout:
        firststep = True
        for _episode in range(100):
            if firststep:
                env.step() # Need to make a first step in order to get an observation.
                firststep = False
            dec, term = env.get_steps(behavior)
            done = False
            episodeReward = 0
            found_reward = False
            step = 0
            while not done:
                reward = None
                if len(dec.reward) > 0:
                    episodeReward += dec.reward
                    reward = dec.reward[0] \
                        if not math.isclose(dec.reward[0], 0, abs_tol=1e-2) else 0
                    if reward > 0:
                        found_reward = True
                    raycasts = env.get_obs_dict(dec.obs)["rays"] # Get the raycast data

                if len(term) > 0:
                    episodeReward += term.reward
                    reward = term.reward[0] \
                        if not math.isclose(term.reward[0], 0, abs_tol=1e-2) else 0
                    if reward > 0:
                        found_reward = True
                    if found_reward:
                        print("found reward")
                    else:
                        print("did not find reward")
                        reward = PUNISHMENT
                    print(F"Episode Reward: {episodeReward}")
                    done = True
                    firststep = True
                    raycasts = env.get_obs_dict(term.obs)["rays"] # Get the raycast data

                stimulus = Stimulus(alearner, alearner.listOfObjects,
                                    raycasts, reward=reward)
                fout.write("Episode %d, step %d:\n" % (_episode+1, step+1))
                fout.write(str(stimulus) + "\n")

                step += 1
                if done:
                    alearner.update_stimulus_values(stimulus)
                    if found_reward:
                        alearner.double_exploit()
                    else:
                        alearner.increase_exploit()
                    alearner.print_maps()
                    break
                # selects action based on stimulus values
                action = alearner.get_action(stimulus)
                env.set_actions(behavior, action.action_tuple)
                env.step()
                dec, term = env.get_steps(behavior)
            env.reset()

    env.close()
    print("Environment Closed")

# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]
    else:
        competition_folder = "../configs/competition/"
        configuration_files = os.listdir(competition_folder)
        configuration_random = random.randint(0, len(configuration_files))
        configuration_file = (
            competition_folder + configuration_files[configuration_random]
        )
    run_agent_single_config(configuration_file=configuration_file)


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
