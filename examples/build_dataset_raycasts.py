from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.raycastparser import RayCastParser
from animalai.envs.raycastparser import RayCastObjects
import sys
import random
import os
import time

TOTAL_RAYS = 9
DATAPOINTS = 10000
OBJECTS = [RayCastObjects.GOODGOAL, RayCastObjects.GOODGOALMULTI, RayCastObjects.BADGOAL, RayCastObjects.ARENA, RayCastObjects.IMMOVABLE, RayCastObjects.MOVABLE]

def load_config_and_play(configuration_file: str) -> None:
    """
    Loads a configuration file for a single arena and lets you play manually
    :param configuration_file: str path to the yaml configuration
    :return: None
    """
    env_path = "../env/AnimalAI"

    seed = int(time.time())
    print("initializing AAI environment")
    env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=configuration_file,
        seed=seed,
        play=False,
        useCamera=True,
        useRayCasts=True,
        raysPerSide=int((TOTAL_RAYS-1)/2),
        rayMaxDegrees=30
    )
    parser = RayCastParser(OBJECTS, TOTAL_RAYS)

    # Run the environment until signal to it is lost
    try:
        for i in range(10):
            behaviour = list(env.behavior_specs.keys())[0]
            env.step()
            dec, term = env.get_steps(behaviour)
            obs = env.get_obs_dict(dec.obs)
            camera = obs["camera"]
            raycasts = obs["rays"]
            parsed_raycasts = parser.parse(raycasts)
            # print(i, camera)
            print(i, parsed_raycasts)
            env.reset()
    finally:
        env.close()


# If an argument is provided then assume it is path to a configuration and use that
# Otherwise load a random competition config.
if __name__ == "__main__":
    configuration_file = "../configs/tests/allobjs-10.yml"
    load_config_and_play(configuration_file=configuration_file)
