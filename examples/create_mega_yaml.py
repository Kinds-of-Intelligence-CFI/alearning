import argparse
import os
import re
import random
import pickle


def _get_arena_contents(directory, in_file):
    content = ""
    with open(os.path.join(directory, in_file), "r") as fin:
        for _ in range(3):
            fin.readline()
        next_line = fin.readline()
        while next_line:
            if re.match("\s*t: [0-9]+", next_line):
                next_line = re.sub("[0-9]+", "2000", next_line)
            content += next_line
            next_line = fin.readline()

    if content[-1] != "\n":
        content += "\n"
    return content


def create_yml(directories, out_dir, out_file, prop):
    print(prop)
    all_tasks = []
    for directory in directories:
        for filename in os.listdir(directory):
            if (
                    filename != "all_tasks.yml"
                    and filename != "meta_data.bin"
                    and filename[-3:] != "csv"
                    and os.path.isfile(os.path.join(directory, filename))
            ):
                all_tasks.append((directory, filename))
    if prop != 1:
        n_select = int(prop * len(all_tasks))
        all_tasks = random.sample(all_tasks, n_select)
    random.shuffle(all_tasks)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, out_file), "w") as fout:
        fout.write("!ArenaConfig\narenas:\n")
        idx = 1

        for directory, task in all_tasks:
            content = _get_arena_contents(directory, task)
            for _ in range(2):
                fout.write("  %d: !Arena\n" % idx)
                fout.write(content)
                idx += 1

    meta_data = []
    for directory, filename in all_tasks:
        components = directory.split("/")
        if components[-1] == "":
            directory = components[-2]
        else:
            directory = components[-1]
        filename = filename.split(".")[0]
        meta_data.append([directory] + filename.split("_")[1:])
    with open(os.path.join(out_dir, "meta_data.bin"), "wb") as fout:
        pickle.dump(meta_data, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Builds complete task type yaml'
    )
    parser.add_argument('directories', type=str, nargs='+')
    parser.add_argument('out_dir', type=str)
    parser.add_argument('prop', type=float)
    parser.add_argument('out_file', type=str, nargs='?',
                        default='all_tasks.yml')

    args = parser.parse_args()

    create_yml(args.directories, args.out_dir, args.out_file, args.prop)
