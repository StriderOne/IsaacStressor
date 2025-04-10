
import math
from isaac_stressor.variations.variation_manager import VariationManagerCfg, VariationCfg
from isaaclab.utils import configclass
from isaac_stressor.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

@configclass
class VariationsCfg:

    no_perturbation = VariationCfg(
        name="no_perturbation",
        func=None,
        mode=None,
        params=None,
    )

    randomize_cabinet_color = VariationCfg(
        name="manip_obj_color",
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "asset_cfg": SceneEntityCfg("cabinet"),
            "mesh_name": ".*/visuals/visuals",
            "event_name": "rep_cube_randomize_color",
        },
    )

    # randomize_cabinet_texture = VariationCfg(
    #     name="manip_obj_tex",
    #     func= mdp.randomize_visual_texture_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet"),
    #         "texture_paths": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    #         ],
    #         "event_name": "object_panel_visual_texture",
    #         "texture_rotation": (math.pi / 2, math.pi / 2),
    #     },
    # )

    # randomize_cabinet_mass = VariationCfg(
    #     name="manip_obj_mass",
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names=["drawer_top"]),
    #         "mass_distribution_params": (0.1, 4.1),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_cabinet_physics_material = VariationCfg(
    #     name="manip_obj_friction",
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet"),
    #         "static_friction_range": (0.5, 1.5),
    #         "dynamic_friction_range": (0.5, 1.5),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

@configclass
class CabinetVariantionsCfg(VariationManagerCfg):

    variations: VariationsCfg = VariationsCfg()
    