# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaacgym import gymtorch

from phys_anim.envs.masked_mimic_inversion.base_task.common import BaseMaskedMimicTask

from phys_anim.envs.masked_mimic_inversion.masked_mimic.masked_mimic_humanoid import (
    MaskedMimicHumanoid,
)


class MaskedMimicTaskHumanoid(BaseMaskedMimicTask, MaskedMimicHumanoid):  # type: ignore[misc]
    def __init__(self, config, device, motion_lib=None):
        config.visualize_markers = False
        super().__init__(config=config, device=device, motion_lib=motion_lib)

        if "smpl" in self.config.robot.asset.asset_file_name:
            self.head_body_id = self.head_id = self.build_body_ids_tensor(["Head"]).item()
        else:
            self.head_body_id = self.head_id = self.build_body_ids_tensor(["head"]).item()

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)
        self.set_perturbations(env_ptr)

        self.build_env_task(env_id, env_ptr, humanoid_asset)

    def set_perturbations(self, env_ptr):
        perturbations = self.config.get("perturbations", {})
        if "friction" in perturbations.keys():
            ground_friction = perturbations["friction"]
            foot_names = ["L_Ankle", "R_Ankle", "L_Toe", "R_Toe"]
            foot_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, 0, name) for name in foot_names]
            rb_shape = self.gym.get_actor_rigid_body_shape_indices(env_ptr, 0)
            rb_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, 0)
            for foot_handle in foot_handles:
                foot_shape = rb_shape[foot_handle]
                rb_shape_props[foot_shape.start].friction = ground_friction
                rb_shape_props[foot_shape.start].rolling_friction = ground_friction
                rb_shape_props[foot_shape.start].torsion_friction = ground_friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, 0, rb_shape_props)
        if "mass_multiplier" in perturbations:
            mass_multiplier = perturbations["mass_multiplier"]
            rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, 0)
            for body_name, multiplier in mass_multiplier.items():
                body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, 0, body_name)
                rb_props[body_handle].mass *= multiplier
            self.gym.set_actor_rigid_body_properties(env_ptr, 0, rb_props)

    def build_env_task(self, env_id, env_ptr, humanoid_asset):
        pass

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_env_tensors(self, env_ids):
        super().reset_env_tensors(env_ids)

        env_ids_int32 = self.get_task_actor_ids_for_reset(env_ids)
        if env_ids_int32 is not None:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    def pre_physics_step(self, actions):
        self.update_task(actions)
        super().pre_physics_step(actions)

    def update_task(self, actions):
        pass

    ###############################################################
    # Getters
    ###############################################################
    def get_task_actor_ids_for_reset(self, env_ids):
        return None

    ###############################################################
    # Helpers
    ###############################################################
    def render(self):
        super().render()

        if self.viewer:
            self.draw_task()
