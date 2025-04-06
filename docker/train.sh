#!/bin/bash

scripts/tools/isaac_stressor_run_docker.sh /workspace/isaaclab/scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Extended-Open-Drawer-Franka-IK-Rel-v0 \
    --algo bc \
    --dataset ./datasets/dataset.hdf5

