import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os


WINDOW_SIZE = 5


def process_results(path, testset_size, output_file):
    directory = os.path.dirname(path)
    results = pd.read_csv(path)
    test_runs = []
    moving_accuracy = []

    n_runs = WINDOW_SIZE
    for i in range(n_runs*testset_size, len(results)+1, testset_size):
        test_runs.append(n_runs)
        mean_acc = results["found_green"][
            (i-WINDOW_SIZE*testset_size):i
        ].mean()
        moving_accuracy.append(mean_acc)
        n_runs += 1

    plt.clf()
    plt.scatter(test_runs, moving_accuracy)
    plt.title("Test set success rate (moving average, window size = %d)"
              % WINDOW_SIZE)
    plt.xlabel("test run")
    plt.ylabel("success rate")
    plt.xlim(left=0)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(directory, output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Processes results'
    )
    parser.add_argument('path', type=str)
    parser.add_argument('testset_size', type=int)
    parser.add_argument('output_file', type=str, nargs='?',
                        default="smoothed_test_results.png")
    args = parser.parse_args()

    process_results(args.path, args.testset_size, args.output_file)
