import argparse
import json
import os

import numpy as np

from utils import nerf_matrix_to_ngp, visualize_poses


def configargperser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms_path", type=str)
    return parser

if __name__ == "__main__":
    parser = configargperser()
    opt = parser.parse_args()

    with open(opt.transforms_path, "r") as f:
        transform = json.load(f)

    frames = transform["frames"]

    poses = []
    for f in frames:
        pose = np.array(f["transform_matrix"], dtype=np.float32)
        pose = nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0])

        poses.append(pose)

    visualize_poses(poses)
