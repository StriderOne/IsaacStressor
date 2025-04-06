#!/bin/bash

./scripts/tools/isaac_stressor_run_docker.sh scripts/tools/record_demos.py \
    --task Isaac-Extended-Open-Drawer-Franka-IK-Rel-v0 \
    --dataset_file ./datasets/dataset.hdf5 \
    --num_demos 10 \
    --enable_cameras \
    --headless \
    --num_envs 25
