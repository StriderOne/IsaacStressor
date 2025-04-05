# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import time
import torch

import omni.log

# from isaaclab.devices import Se3HandTracking, Se3Keyboard, Se3SpaceMouse
# from isaaclab.envs import ViewerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
# from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
# from isaaclab.envs.ui import ViewportCameraController
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from scipy.spatial.transform import Rotation

def substract_poses(pose_tensor, goal_tensor):
    pose_positions = pose_tensor[:3]  # [x, y, z]
    pose_quaternions = pose_tensor[3:]  # [qw, qx, qy, qz]
    
    # Reorder quaternion to SciPy format: [qx, qy, qz, qw]
    pose_quaternions_reordered = torch.cat([
        pose_quaternions[1:],  # [qx, qy, qz]
        pose_quaternions[:1]    # [qw]
    ], dim=0)
    r1 = Rotation.from_quat(pose_quaternions_reordered.cpu().numpy()).as_matrix()
    goal_positions = goal_tensor[:3]  # [x, y, z]
    goal_quaternions = goal_tensor[3:]  # [qw, qx, qy, qz]
    
    # Reorder quaternion to SciPy format: [qx, qy, qz, qw]
    goal_quaternions_reordered = torch.cat([
        goal_quaternions[1:],  # [qx, qy, qz]
        goal_quaternions[:1]    # [qw]
    ], dim=0)
    r2 = Rotation.from_quat(goal_quaternions_reordered.cpu().numpy()).as_matrix()
    angles = r2 @ r1.T
    rotvecs = Rotation.from_matrix(angles).as_rotvec()
    # Combine and return on the same device
    return torch.cat([
        (goal_positions - pose_positions).unsqueeze(0),
        torch.tensor(rotvecs, device=pose_tensor.device, dtype=pose_tensor.dtype).unsqueeze(0)
    ], dim=1)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    should_reset_recording_instance = False
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
    
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(env.scene)
    robot = env.scene["robot"]
    cabinet_data = env.scene["cabinet_frame"].data
    # Define goals for the arm
    # ee_goals = [
    #     [0.22, 0, 0.72, 0.5, 0.5, 0.5, 0.5],
    #     [0.36, 0, 0.72, 0.5, 0.5, 0.5, 0.5],
    #     [0.36, 0, 0.72, 0.5, 0.5, 0.5, 0.5],
    #     [0.14, 0, 0.72, 0.5, 0.5, 0.5, 0.5]
    # ]
    ee_goals = [
        [0., 0., -0.22, 1., 0., 0., 0.],
        [0., 0., -0.08, 1., 0., 0., 0.],
        [0., 0., -0.08, 1., 0., 0., 0.],
        [0., 0., -0.32, 1., 0., 0., 0.]
    ]
    ee_goals = torch.tensor(ee_goals, device=env.sim.device)
    
    # print("EE goals in world frame:", ee_goals_pos_w, ee_goals_quat_w)
    gripper_goals = [
        0, 0, 1, 1
    ]
    
    gripper_goals = torch.tensor(gripper_goals, device=env.sim.device)
    # Track the given command
    current_goal_idx = 0

    # reset before starting
    env.reset()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
           
            # get keyboard command
            gripper_command = gripper_goals[current_goal_idx]
            # compute actions based on environment
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

            ee_goals_pos_w, ee_goals_quat_w = combine_frame_transforms(
                cabinet_data.target_pos_w.squeeze(1), cabinet_data.target_quat_w.squeeze(1), ee_goals[current_goal_idx, 0:3].unsqueeze(0), ee_goals[current_goal_idx, 3:7].unsqueeze(0)
            )
            ee_goals_w = torch.cat([ee_goals_pos_w, ee_goals_quat_w], dim=1).squeeze(0)
            
            # hardcode to make accleration of eef on the last task smaller, otherwise it is too fast and the handle slips out of the gripper
            gain = 1.0
            if current_goal_idx == 3:
                gain = 0.5
            actions = pre_process_actions(gain*substract_poses(ee_pose_w[0, :], ee_goals_w), gripper_command)

            # perform action on environment
            obs, rew, terminated, truncated, info = env.step(actions)

            # check if current goal has been reached
            if torch.allclose(ee_pose_w[0, :3], ee_goals_w[:3], atol=1e-2):
                
                if (gripper_goals[current_goal_idx] == 1 and gripper_goals[current_goal_idx - 1] == 0):
                    if torch.all(torch.abs(obs['policy']['joint_vel'][:, -2:]) < 0.01):
                        current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
                else:
                    current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
         
            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0

            if should_reset_recording_instance:
                env.recorder_manager.reset()
                current_goal_idx = 0
                env.reset()
                print(f"Cabinet data: {cabinet_data.target_pos_w, cabinet_data.target_quat_w}")
    
                should_reset_recording_instance = False
                success_step_count = 0

            # print out the current demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
