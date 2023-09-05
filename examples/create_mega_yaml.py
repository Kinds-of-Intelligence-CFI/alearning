import argparse
import os
import re


def _get_arena_contents(directory, in_file):
    content = ""
    with open(os.path.join(directory, in_file), "r") as fin:
        for _ in range(3):
            fin.readline()
        next_line = fin.readline()
        while next_line:
            if re.match("\s*t: [0-9]+", next_line):
                next_line = re.sub("[0-9]+", "1000", next_line)
            content += next_line
            next_line = fin.readline()

    if content[-1] != "\n":
        content += "\n"
    return content

def create_yml(directory, out_file):
    all_tasks = []
    for filename in os.listdir(directory):
        if (
                filename != "all_tasks.yml"
                and os.path.isfile(os.path.join(directory, filename))
        ):
            all_tasks.append(filename)

    with open(directory + "/" + out_file, "w") as fout:
        fout.write("!ArenaConfig\narenas:\n")
        idx = 1

        for task in all_tasks:
            content = _get_arena_contents(directory, task)
            for _ in range(2):
                fout.write("  %d: !Arena\n" % idx)
                fout.write(content)
                idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Builds complete task type yaml'
    )
    parser.add_argument('directory', type=str)
    parser.add_argument('out_file', type=str, nargs='?',
                        default='all_tasks.yml')

    args = parser.parse_args()

    create_yml(args.directory, args.out_file)
