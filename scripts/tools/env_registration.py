import gymnasium as gym

gym.register(
    id="Isaac-Extended-Open-Drawer-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaac_stressor.tasks.manager_based.cabinet.config.franka.ik_rel_env_cfg:FrankaCabinetEnvCfg",
        "robomimic_bc_cfg_entry_point": "source/isaac_stressor/isaac_stressor/tasks/manager_based/cabinet/config/franka/agents/robomimic/bc_rnn_low_dim.json",

    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Extended-Cabinet-Franka-IK-Rel-Mimic-v0",
    entry_point="isaac_stressor.envs:FrankaCabinetIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": "isaac_stressor.envs.franka_cabinet_ik_rel_mimic_env_cfg:FrankaCabinetIKRelMimicEnvCfg",
    },
    disable_env_checker=True,
)
