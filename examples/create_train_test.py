import os


def _get_arena_contents(directory, in_file):
    content = ""
    with open(os.path.join(directory, in_file), "r") as fin:
        for _ in range(3):
            fin.readline()
        next_line = fin.readline()
        while next_line:
            content += next_line
            next_line = fin.readline()

    if content[-1] != "\n":
        content += "\n"
    return content

def create_yml(directories, out_path):
    with open(out_path, "w") as fout:
        fout.write("!ArenaConfig\narenas:\n")
        idx = 1

        for directory in directories:
            all_tasks = []
            for filename in os.listdir(directory):
                if (
                        os.path.isfile(os.path.join(directory, filename))
                        and filename != "all_tasks.yml"
                ):
                    all_tasks.append(filename)

            for task in all_tasks:
                content = _get_arena_contents(directory, task)
                for _ in range(2):
                    fout.write("  %d: !Arena\n" % idx)
                    fout.write(content)
                    idx += 1


if __name__ == "__main__":
    training_directories = [
        "../configs/basic_curriculum/task_type_2_GR_FA",
        "../configs/basic_curriculum/task_type_3_GR_Platform_HC"
    ]

    testing_directories = [
        "../configs/basic_curriculum/task_type_1_GR_HC",
        "../configs/basic_curriculum/task_type_4_GR_Platform_FA"
    ]

    if not os.path.exists("../configs/autoencoder"):
        os.makedirs("../configs/autoencoder")

    create_yml(training_directories, "../configs/autoencoder/training.yml")
    create_yml(testing_directories, "../configs/autoencoder/testing.yml")
