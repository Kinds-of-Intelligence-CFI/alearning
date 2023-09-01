from animalai.envs.environment import AnimalAIEnvironment
from gym_unity.envs import UnityToGymWrapper
import sys
import time
import numpy as np
import pickle
import random

N_TRAIN_DATAPOINTS = 100000
N_TEST_DATAPOINTS = 10000


def load_config_and_build_dataset(train_configuration_file: str,
                                  test_configuration_file: str,
                                  train_file: str, test_file: str) -> None:
    """
    Load a configuration file and build dataset for autoencoder.

    :param configuration_file: str path to the yaml configuration
    :return: None
    """
    env_path = "../env/AnimalAI"

    seed = int(time.time())
    print("initializing AAI environment for training data")
    env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=train_configuration_file,
        seed=seed,
        play=False,
        useCamera=True,
        useRayCasts=False
    )
    env = UnityToGymWrapper(env, uint8_visual=True, flatten_branched=True)

    train_data = []
    reset = False
    try:
        for i in range(N_TRAIN_DATAPOINTS):
            if reset:
                print("Resetting environment...")
                env.reset()
                reset = False
            action = random.randrange(env.action_space.n)
            res = env.step(action)
            train_data.append(res[0])
            if (i+1) % 1000 == 0:
                print("Number of datapoints: %d" % (i+1))
            if res[2]:
                reset = True
    finally:
        env.close()

    seed = int(time.time())
    print("initializing AAI environment for testing data")
    env = AnimalAIEnvironment(
        file_name=env_path,
        arenas_configurations=test_configuration_file,
        seed=seed,
        play=False,
        useCamera=True,
        useRayCasts=False
    )
    env = UnityToGymWrapper(env, uint8_visual=True, flatten_branched=True)
    test_data = []
    reset = False
    try:
        for i in range(N_TEST_DATAPOINTS):
            if reset:
                print("Resetting environment...")
                env.reset()
                reset = False
            action = random.randrange(env.action_space.n)
            res = env.step(action)
            test_data.append(res[0])
            if (i+1) % 1000 == 0:
                print("Number of datapoints: %d" % (i+1))
            if res[2]:
                reset = True
    finally:
        env.close()

    print("Writing to files...")
    train_data = np.array(train_data, dtype=np.uint8)
    np.random.shuffle(train_data)
    test_data = np.array(test_data, dtype=np.uint8)
    np.random.shuffle(test_data)
    with open(train_file, "wb") as fout:
        pickle.dump(train_data, fout, protocol=4)
    with open(test_file, "wb") as fout:
        pickle.dump(test_data, fout, protocol=4)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_file = "train_data.bin"
        test_file = "test_data.bin"
    elif len(sys.argv) == 3:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
    else:
        sys.stderr.write("Wrong number of arguments provided\n")
        sys.exit(1)
    train_configuration_file = "../configs/autoencoder/training.yml"
    test_configuration_file = "../configs/autoencoder/testing.yml"
    load_config_and_build_dataset(
        train_configuration_file=train_configuration_file,
        test_configuration_file=test_configuration_file,
        train_file=train_file, test_file=test_file
    )
