# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaac_stressor.tasks.manager_based.cabinet import mdp

from isaac_stressor.tasks.manager_based.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

##
# Pre-defined configs
##
from isaac_stressor.assets.robots.arx5 import ARX5_CFG  # isort: skip

from isaac_stressor.tasks.manager_based.rack.rack_env_cfg import RackEnvCfg

@configclass
class ARX5RackEnvCfg(RackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        # self.scene.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        #     debug_vis=False,
        #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        #             name="ee_tcp",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.1034),
        #             ),
        #         ),
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        #             name="tool_leftfinger",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.046),
        #             ),
        #         ),
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        #             name="tool_rightfinger",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.046),
        #             ),
        #         ),
        #     ],
        # )

        # override rewards
        # self.rewards.approach_gripper_handle.params["offset"] = 0.04
        # self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        # self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger_.*"]


@configclass
class ARX5RackEnvCfg_PLAY(RackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
