import argparse
import os


def _get_arena_contents(directory, in_file):
    content = ""
    with open(directory + "/" + in_file, "r") as fin:
        for _ in range(3):
            fin.readline()
        next_line = fin.readline()
        while next_line:
            content += next_line
            next_line = fin.readline()

    if content[-1] != "\n":
        content += "\n"
    return content

def create_yml(directory, n_reps, out_file):
    all_files = [f for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))
                 and f != out_file]
    with open(directory + "/" + out_file, "w") as fout:
        fout.write("!ArenaConfig\narenas:\n")
        idx = 1

        for filename in all_files:
            content = _get_arena_contents(directory, filename)
            for _ in range(n_reps):
                for _ in range(2):
                    fout.write("  %d: !Arena\n" % idx)
                    fout.write(content)
                    idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Builds training yaml from a set of files'
    )
    parser.add_argument('directory', type=str)
    parser.add_argument('n_reps', type=int)
    parser.add_argument('out_file', type=str, nargs='?',
                        default='basic_curriculum.yml')

    args = parser.parse_args()

    create_yml(args.directory, args.n_reps, args.out_file)
