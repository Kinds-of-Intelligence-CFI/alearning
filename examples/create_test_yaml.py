import argparse
import os
import re
import random
import pickle
from create_mega_yaml import _get_arena_contents, _get_tasks, _parse_filename


def create_test_yml(old_directories, directories, out_dir,
                    out_file, n_steps, prop_old=0.2, prop=1):
    old_tasks = _get_tasks(old_directories)
    if prop_old < 1:
        n_select = int(prop_old * len(old_tasks))
        old_tasks = random.sample(old_tasks, n_select)

    all_tasks = _get_tasks(directories)
    if prop < 1:
        n_select = int(prop * len(all_tasks))
        all_tasks = random.sample(all_tasks, n_select)

    all_tasks.extend(old_tasks)
    random.shuffle(all_tasks)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, out_file), "w") as fout:
        fout.write("!ArenaConfig\narenas:\n")
        idx = 0

        for directory, task in all_tasks:
            content = _get_arena_contents(directory, task, n_steps)
            fout.write("  %d: !Arena\n" % idx)
            fout.write(content)
            idx += 1

    meta_data = [len(all_tasks)]
    for directory, filename in all_tasks:
        components = directory.split("/")
        if components[-1] == "":
            directory = components[-2]
        else:
            directory = components[-1]
        filename = filename.split(".")[0]
        norm_args = _parse_filename(directory, filename)
        meta_data.append([directory] + norm_args)
    with open(os.path.join(out_dir, "meta_data.bin"), "wb") as fout:
        pickle.dump(meta_data, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Builds complete task type yaml'
    )
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_steps', type=int)
    parser.add_argument('prop', type=float, nargs='?', default=1)
    parser.add_argument('prop_old', type=float, nargs='?', default=0.2)
    parser.add_argument('--dirs', type=str, nargs='+')
    parser.add_argument('--old_dirs', type=str, nargs='*', default=[])
    parser.add_argument('--out', type=str, nargs='?',
                        default='all_tasks.yml')

    args = parser.parse_args()

    create_test_yml(args.old_dirs, args.dirs,
                    args.out_dir, args.out,
                    args.n_steps,
                    prop_old=args.prop_old, prop=args.prop)
