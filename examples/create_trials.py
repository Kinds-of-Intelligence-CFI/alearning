import argparse
import random


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

def create_yml(directory, out_file):
    with open(directory + "/" + out_file, "w") as fout:
        fout.write("!ArenaConfig\narenas:\n")
        idx = 1

        # stage 1
        content = _get_arena_contents(directory, "L-G.yml")
        for _ in range(10):
            for _ in range(2):
                fout.write("  %d: !Arena\n" % idx)
                fout.write(content)
                idx += 1

        content = _get_arena_contents(directory, "R-G.yml")
        for _ in range(10):
            for _ in range(2):
                fout.write("  %d: !Arena\n" % idx)
                fout.write(content)
                idx += 1

        content = _get_arena_contents(directory, "L-G-R-G-P-GR.yml")
        for _ in range(20):
            for _ in range(2):
                fout.write("  %d: !Arena\n" % idx)
                fout.write(content)
                idx += 1

        # stage 3
        for _ in range(15):
            previous = -1
            choose_random = True
            test_files = []
            contents = list(map(lambda x: _get_arena_contents(directory, x),
                                ["L-G-R-L-P-Y.yml", "L-L-R-G-P-B.yml"]))
            for _ in range(20):
                if not choose_random:
                    choice = 0 if previous == 1 else 1
                    choose_random = True
                else:
                    choice = random.randrange(0, 2)

                test_files.append(choice)
                if choice == previous:
                    choose_random = False
                previous = choice

            for file_id in test_files:
                for _ in range(2):
                    fout.write("  %d: !Arena\n" % idx)
                    fout.write(contents[file_id])
                    idx += 1

            previous = -1
            choose_random = True
            test_files = []
            contents = list(map(lambda x: _get_arena_contents(directory, x),
                                ["L-G-R-L-P-C.yml", "L-L-R-G-P-CR.yml"]))
            for _ in range(20):
                if not choose_random:
                    choice = 0 if previous == 1 else 1
                    choose_random = True
                else:
                    choice = random.randrange(0, 2)

                test_files.append(choice)
                if choice == previous:
                    choose_random = False
                previous = choice

            for file_id in test_files:
                for _ in range(2):
                    fout.write("  %d: !Arena\n" % idx)
                    fout.write(contents[file_id])
                    idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Builds complete training yaml'
    )
    parser.add_argument('directory', type=str)
    parser.add_argument('out_file', type=str, nargs='?',
                        default='mondragon.yml')

    args = parser.parse_args()

    create_yml(args.directory, args.out_file)
