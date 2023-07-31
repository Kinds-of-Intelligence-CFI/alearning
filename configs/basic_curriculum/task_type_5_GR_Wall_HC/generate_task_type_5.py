#Here's the original with some additional comments made.
import itertools
import os
import argparse

#Template for the config. \n means to generate a line break and %d is to insert a number, while %s is to insert a string.
def generate_arena_content(goal_size, x_pos, goal_dist_x_idx,
                           z_pos, goal_dist_z_idx):
    content = "" \
        "!ArenaConfig\n" \
        "arenas:\n" \
        "  0: !Arena\n" \
        "    pass_mark: 0\n" \
        "    t: 100\n" \
        "    items:\n" \
        "    - !Item\n" \
        "      name: Agent\n" \
        "      positions:\n" \
        "      - !Vector3 {x: %d, y: 1, z: %d}\n" \
        "      rotations: [0]\n" \
        "    - !Item\n" \
        "      name: Wall\n" \
        "      positions:\n" \
        "      - !Vector3 {x: 20, y: 0, z: 26}\n" \
        "      - !Vector3 {x: 20, y: 0, z: 14}\n" \
        "      - !Vector3 {x: 26, y: 0, z: 20}\n" \
        "      - !Vector3 {x: 14, y: 0, z: 20}\n" \
        "      rotations: [0, 0, 0, 0]\n" \
        "      colors:\n" \
        "      - !RGB {r: 153, g: 153, b: 153}\n" \
        "      - !RGB {r: 153, g: 153, b: 153}\n" \
        "      - !RGB {r: 153, g: 153, b: 153}\n" \
        "      - !RGB {r: 153, g: 153, b: 153}\n" \
        "      sizes:\n" \
        "      - !Vector3 {x: 13, y: 2, z: 1}\n" \
        "      - !Vector3 {x: 13, y: 2, z: 1}\n" \
        "      - !Vector3 {x: 1, y: 2, z: 11}\n" \
        "      - !Vector3 {x: 1, y: 2, z: 11}\n" \
        "    - !Item\n" \
        "      name: Wall\n" \
        "      positions:\n" \
        "      - !Vector3 {x: 24, y: 0, z: 22}\n" \
        "      rotations: [0]\n" \
        "      colors:\n" \
        "      - !RGB {r: 153, g: 153, b: 153}\n" \
        "      sizes:\n" \
        "      - !Vector3 {x: 1, y: 2, z: 5}\n" \
        "    - !Item\n" \
        "      name: GoodGoal\n" \
        "      positions:\n" \
        "      - !Vector3 {x: %d, y: 1, z: %d}\n" \
        "      sizes:\n" \
        "      - !Vector3 %s\n" % (
            x_pos[0],
            z_pos[0],
            x_pos[1][goal_dist_x_idx],
            z_pos[1][goal_dist_z_idx],
            goal_size
        )

    return content
# wall sizes should have values of
# & 1-5.
#The string is a key that is used to call the object, in this case a 3 item tuple.
def main(out_dir):
    goal_sizes = {
        "large": "{x: 2, y: 2, z: 2}",
        "medium": "{x: 1, y: 1, z: 1}",
        "small": "{x: 0.5, y: 0.5, z: 0.5}"
    }

    x_pos = {
        "right": (15, (17, 21, 24)),
        "left": (25, (23, 21, 16))
    }

    z_pos = {
        "forward": (15, (17, 21, 24)),
        "behind": (25, (23, 21, 16))
    }

    goal_dist_idx = {
        "close": 0,
        "medium": 1,
        "far": 2
    }

#Change out_dir to the file directory desired.
    if not os.path.exists(path= r"C:\Users\User\Documents\Doctorate documents\animal-ai-3.0.2\configs\competition\Custom\basic curriculum"):
        os.makedirs(path= r"C:\Users\User\Documents\Doctorate documents\animal-ai-3.0.2\configs\competition\Custom\basic curriculum")

    counter = 0
    for size, x_pos_k, dist_x, z_pos_k, dist_z \
        in itertools.product(goal_sizes.keys(), x_pos.keys(),
                             goal_dist_idx.keys(), z_pos.keys(),
                             goal_dist_idx.keys()):
        filename = r"C:\Users\User\Documents\Doctorate documents\animal-ai-3.0.2\configs\competition\Custom\basic curriculum" + "/%05d_%s_%s_%s_%s_%s_wall_1x5_right_upper.yaml" \
            % (counter, size, x_pos_k, dist_x, z_pos_k, dist_z)
        with open(filename, "w") as fout:
            content = generate_arena_content(goal_sizes[size], x_pos[x_pos_k],
                                             goal_dist_idx[dist_x],
                                             z_pos[z_pos_k],
                                             goal_dist_idx[dist_z])
            fout.write(content)
        counter += 1


# the __name__ == __main__ is not important here but is used in another coding to restrict the files that are run to only the one that in on the main path.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates configs for task type 1 of the basic curriculum'
    )
    parser.add_argument('out_dir', type=str, nargs='?',
                        default='./task_type_3_GR_HC/')
    args = parser.parse_args()

    main(args.out_dir)
