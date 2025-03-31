# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .franka_cabinet_ik_rel_mimic_env import FrankaCabinetIKRelMimicEnv
from .franka_cabinet_ik_rel_mimic_env_cfg import FrankaCabinetIKRelMimicEnvCfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Cabinet-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCabinetIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_cabinet_ik_rel_mimic_env_cfg.FrankaCabinetIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)
