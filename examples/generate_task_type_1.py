import itertools
import os
import argparse


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
        "      - !Vector3 {x: %d, y: 0, z: %d}\n" \
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
        "      name: GoodGoal\n" \
        "      positions:\n" \
        "      - !Vector3 {x: %d, y: 0, z: %d}\n" \
        "      sizes:\n" \
        "      - !Vector3 %s\n" % (
            x_pos[0],
            z_pos[0],
            x_pos[1][goal_dist_x_idx],
            z_pos[1][goal_dist_z_idx],
            goal_size
        )

    return content


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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    counter = 0
    for size, x_pos_k, dist_x, z_pos_k, dist_z \
        in itertools.product(goal_sizes.keys(), x_pos.keys(),
                             goal_dist_idx.keys(), z_pos.keys(),
                             goal_dist_idx.keys()):
        filename = out_dir + "/%05d_%s_%s_%s_%s_%s.yaml" \
            % (counter, size, x_pos_k, dist_x, z_pos_k, dist_z)
        with open(filename, "w") as fout:
            content = generate_arena_content(goal_sizes[size], x_pos[x_pos_k],
                                             goal_dist_idx[dist_x],
                                             z_pos[z_pos_k],
                                             goal_dist_idx[dist_z])
            fout.write(content)
        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates configs for task type 1 of the basic curriculum'
    )
    parser.add_argument('out_dir', type=str, nargs='?',
                        default='./task_type_1_GR_HC/')
    args = parser.parse_args()

    main(args.out_dir)
