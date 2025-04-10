# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--variation", type=int, default=0, help="Random seed.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from isaaclab_tasks.utils import parse_env_cfg
import isaac_stressor
from isaac_stressor.variations.variation_manager import VariationManager
from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.scene import InteractiveScene

import importlib
from omegaconf import DictConfig

def get_variation_manager_cfg(env_name: str) -> DictConfig:
    # Get the gym registration entry
    gym_spec = gym.spec(env_name)
    if gym_spec is None:
        raise ValueError(f"Environment {env_name} is not registered")
    
    # Get the entry point path
    cfg_entry_point = gym_spec.kwargs["variation_manager_cfg_entry_point"]
    
    # Split into module and class name
    module_path, class_name = cfg_entry_point.split(":")
    
    # Import the module
    module = importlib.import_module(module_path)
    
    # Get the class
    cfg_class = getattr(module, class_name)
    
    # Create and return the config object
    return cfg_class()

def rollout(policy, env, horizon, device):
    policy.start_episode
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for i in range(horizon):
        # Prepare observations
        obs = obs_dict["policy"]
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        traj["obs"].append(obs)
        #TODO
        # obs["table_cam"] = obs["table_cam"].permute(2, 0, 1)    # change to (C, H, W) for inference
    
        # Compute actions
        actions = policy(obs)
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        if terminated:
            return True, traj
        elif truncated:
            return False, traj

    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    variation_manager_cfg = get_variation_manager_cfg(args_cli.task)
    variation_manager = VariationManager(variation_manager_cfg)

    # print(len(variation_manager.generate_config(env_cfg, variations_cfg)))
    modified_env_cfg = variation_manager.generate_env_config(env_cfg, args_cli.variation)
    env = gym.make(args_cli.task, cfg=modified_env_cfg).unwrapped
    # env.scene.cabinet = InteractiveScene(env_cfg.scene)
    # env.event_manager = EventManager(modified_env_cfg.events, env)
    # env.reset()
    # env.sim.reset()
    # Create environment
    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Load policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        terminated, traj = rollout(policy, env, args_cli.horizon, device)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()

    
        
   
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
