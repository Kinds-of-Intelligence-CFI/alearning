import argparse
import os
import re
import random
import pickle


def _get_arena_contents(directory, in_file, n_steps):
    content = ""
    with open(os.path.join(directory, in_file), "r") as fin:
        for _ in range(3):
            fin.readline()
        next_line = fin.readline()
        while next_line:
            if re.match("\s*t: [0-9]+", next_line):
                next_line = re.sub("[0-9]+", ("%d" % n_steps), next_line)
            content += next_line
            next_line = fin.readline()

    if content[-1] != "\n":
        content += "\n"
    return content


def _get_tasks(directories):
    tasks = []
    for directory in directories:
        for filename in os.listdir(directory):
            if (
                    filename != "all_tasks.yml"
                    and filename != "meta_data.bin"
                    and filename[-3:] != "csv"
                    and filename[-3:] != "ini"
                    and os.path.isfile(os.path.join(directory, filename))
            ):
                tasks.append((directory, filename))
    return tasks


def _parse_filename(dir_name, filename):
    args = filename.split("_")
    args = [arg for arg in args if arg != ""]
    if dir_name == "base_training":
        return args

    task_type = int(dir_name.split("_")[2])
    norm_args = []

    size_dirname = len(dir_name.split("_"))
    if task_type == 1 or task_type == 2:
        norm_args.extend(args[:5])
        norm_args.extend(["n/a"] * 12)
        norm_args.extend(args[5:])
    elif task_type == 3 or task_type == 4:
        if len(args) > 11:
            args = [arg for arg in args if arg != "far"]
        norm_args.extend(args[:9])
        norm_args.extend(["n/a"] * 8)
        norm_args.extend(args[9:])
    elif task_type == 5:
        if size_dirname == 7:
            norm_args.extend(args[:9])
            norm_args.extend(["n/a"] * 8)
            norm_args.extend(args[9:])
        else:
            norm_args.extend(args[:13])
            norm_args.extend(["n/a"] * 4)
            norm_args.extend(args[13:])
    elif task_type == 6:
        if size_dirname == 9:
            norm_args = args
        elif size_dirname == 8:
            norm_args.extend(args[:13])
            norm_args.extend(["n/a"] * 4)
            norm_args.extend(args[13:])
        else:
            norm_args.extend(args[:9])
            norm_args.extend(["n/a"] * 8)
            norm_args.extend(args[9:])

    return norm_args


def create_ymls(old_directories, directories, out_dir, out_file,
                n_steps, prop_old=0.05, prop=1, prop_test=0.1):
    test_freqs = [2, 5, 10]
    old_tasks = _get_tasks(old_directories)
    if prop_old < 1:
        n_select = int(prop_old * len(old_tasks))
        old_tasks = random.sample(old_tasks, n_select)

    all_tasks = _get_tasks(directories)
    if prop < 1:
        n_select = int(prop * len(all_tasks))
        all_tasks = random.sample(all_tasks, n_select)

    n_test = int(prop_test * len(all_tasks))
    test_tasks = random.sample(all_tasks, n_test)
    train_tasks = [t for t in all_tasks if t not in test_tasks]

    train_tasks += old_tasks

    random.shuffle(train_tasks)
    random.shuffle(test_tasks)

    for test_freq in test_freqs:
        norm_out_dir = os.path.join(out_dir, "test_freq=%d" % test_freq)
        create_yml(train_tasks, test_tasks, norm_out_dir,
                   out_file, n_steps, test_freq=test_freq)


def create_yml(train_tasks, test_tasks,
               out_dir, out_file,
               n_steps, test_freq=10):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    total_tasks = []
    with open(os.path.join(out_dir, out_file), "w") as fout:
        fout.write("!ArenaConfig\narenas:\n")
        idx = 0

        for i, (train_dir, train_task) in enumerate(train_tasks):
            if i % test_freq == 0:
                for directory, task in test_tasks:
                    content = _get_arena_contents(directory, task, n_steps)
                    fout.write("  %d: !Arena\n" % idx)
                    fout.write(content)
                    total_tasks.append((directory, task))
                    idx += 1
            content = _get_arena_contents(train_dir, train_task, n_steps)
            fout.write("  %d: !Arena\n" % idx)
            fout.write(content)
            total_tasks.append((train_dir, train_task))
            idx += 1

        for directory, task in test_tasks:
            content = _get_arena_contents(directory, task, n_steps)
            fout.write("  %d: !Arena\n" % idx)
            fout.write(content)
            total_tasks.append((directory, task))
            idx += 1

    meta_data = [len(train_tasks), len(test_tasks), len(total_tasks)]
    for directory, filename in total_tasks:
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
    parser.add_argument('prop_old', type=float, nargs='?', default=0.05)
    parser.add_argument('--dirs', type=str, nargs='+')
    parser.add_argument('--old_dirs', type=str, nargs='*', default=[])
    parser.add_argument('--out', type=str, nargs='?',
                        default='all_tasks.yml')

    args = parser.parse_args()

    create_ymls(args.old_dirs, args.dirs,
                args.out_dir, args.out,
                args.n_steps,
                prop_old=args.prop_old, prop=args.prop)
