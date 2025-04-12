from typing import Optional

import numpy as np
from isaacgym import gymapi, gymtorch
import torch

from isaac_utils import torch_utils
from phys_anim.envs.masked_mimic.common import BaseMaskedMimic
from phys_anim.envs.mimic.isaacgym import MimicHumanoid
from phys_anim.utils.motion_lib import MotionLib


class MaskedMimicHumanoid(BaseMaskedMimic, MimicHumanoid):  # type: ignore[misc]
    def __init__(
        self, config, device: torch.device, motion_lib: Optional[MotionLib] = None
    ):
        super().__init__(config, device)

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        num_markers_per_env = len(self.config.masked_mimic_conditionable_bodies)
        for i in range(num_markers_per_env):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            color = gymapi.Vec3(0.8, 0.0, 0.0)
            self.gym.set_rigid_body_color(
                env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
            )
            self._marker_handles[env_id].append(marker_handle)

        for i in range(num_markers_per_env):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            color = gymapi.Vec3(1.0, 1.0, 0.0)
            self.gym.set_rigid_body_color(
                env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
            )
            self._marker_handles[env_id].append(marker_handle)

        if self.terrain is not None:
            for i in range(self.num_height_points):
                marker_handle = self.gym.create_actor(
                    env_ptr,
                    self._marker_asset,
                    default_pose,
                    "marker",
                    self.num_envs + 10,
                    0,
                    0,
                )
                color = gymapi.Vec3(0.0, 0.8, 0.0)
                self.gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
                )
                self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_markers_per_env = len(self.config.masked_mimic_conditionable_bodies) * 2
        if self.terrain is not None:
            num_markers_per_env += self.num_height_points

        num_actors = self.get_num_actors_per_env()
        if self.total_num_objects > 0:
            self._marker_states = self.root_states[: -self.total_num_objects].view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., 1 : (1 + num_markers_per_env), :]
        else:
            self._marker_states = self.root_states.view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., 1 : (1 + num_markers_per_env), :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self.humanoid_actor_ids.unsqueeze(
            -1
        ) + torch_utils.to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        # Standard future poses
        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        target_pos = ref_state.rb_pos
        target_pos += self.respawn_offsets.clone().view(self.num_envs, 1, 3)

        env_global_positions = self.convert_to_global_coords(
            target_pos[:, 0, :2], self.env_offsets[..., :2]
        )
        target_pos[..., -1:] += self.get_ground_heights(env_global_positions).view(
            self.num_envs, 1, 1
        )

        target_pos = target_pos[:, self.masked_mimic_conditionable_bodies_ids, :]

        inactive_markers = torch.ones(
            self.num_envs,
            len(self.config.masked_mimic_conditionable_bodies),
            dtype=torch.bool,
            device=self.device,
        )

        if self.config.get("masked_mimic_time_mask", False):
            mask_time_len = self.config.num_future_steps
        else:
            mask_time_len = 1

        translation_view = self.masked_mimic_target_bodies_masks.view(
            self.num_envs, mask_time_len, self.num_conditionable_bodies, 2
        )[
            :, 0, :-1, 0
        ]  # ignore the last entry, that is for speed/heading
        active_translations = translation_view == 1

        inactive_markers[active_translations] = False

        target_pos[inactive_markers] += 100

        self._marker_pos[:, : self.masked_mimic_conditionable_bodies_ids.shape[0]] = (
            target_pos
        )

        # Inbetweening target pose
        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.target_pose_time
        )
        target_pos = ref_state.rb_pos
        target_pos += self.respawn_offsets.clone().view(self.num_envs, 1, 3)
        env_global_positions = self.convert_to_global_coords(
            target_pos[:, 0, :2], self.env_offsets[..., :2]
        )
        target_pos[..., -1:] += self.get_ground_heights(env_global_positions).view(
            self.num_envs, 1, 1
        )

        target_pos = target_pos[:, self.masked_mimic_conditionable_bodies_ids, :]

        translation_view = self.target_pose_joints.view(
            self.num_envs, self.num_conditionable_bodies, 2
        )[
            :, :-1, 0
        ]  # ignore the last entry, that is for speed/heading
        active_translations = translation_view == 1

        inactive_markers[active_translations] = False

        target_pos[inactive_markers] += 100

        target_pos[torch.logical_not(self.target_pose_obs_mask.view(-1))] += 100
        self._marker_pos[
            :,
            self.masked_mimic_conditionable_bodies_ids.shape[
                0
            ] : self.masked_mimic_conditionable_bodies_ids.shape[0]
            * 2,
        ] = target_pos

        # Terrain
        if self.terrain is not None:
            height_maps = self.get_height_maps(None, return_all_dims=True)
            height_maps = height_maps.view(self.num_envs, -1, 3)
            height_maps[..., :2] -= self.env_offsets[..., :2].view(self.num_envs, 1, 2)
            self._marker_pos[
                :, self.masked_mimic_conditionable_bodies_ids.shape[0] * 2 :
            ] = height_maps

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

    def draw_mimic_markers(self):
        self._update_marker()

        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        target_rot = ref_state.rb_rot
        current_state = self.get_bodies_state()
        current_pos, current_rot = (
            current_state.body_pos,
            current_state.body_rot,
        )

        target_rot = target_rot[:, self.masked_mimic_conditionable_bodies_ids, :]
        current_pos = current_pos[:, self.masked_mimic_conditionable_bodies_ids, :]
        current_rot = current_rot[:, self.masked_mimic_conditionable_bodies_ids, :]

        rotation_cols = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        if self.config.masked_mimic_time_mask:
            mask_time_len = self.config.num_future_steps
        else:
            mask_time_len = 1

        # TODO: also plot future poses when using time-based mask?
        rotation_view = self.masked_mimic_target_bodies_masks.view(
            self.num_envs, mask_time_len, self.num_conditionable_bodies, 2
        )[
            :, 0, :-1, 1
        ]  # remove last entry for speed/heading
        active_rotations = rotation_view == 1

        for i, env_ptr in enumerate(self.envs):
            env_active_rotations = active_rotations[i]

            env_current_pos = current_pos[i, env_active_rotations]

            env_target_rot = target_rot[i, env_active_rotations]
            env_current_rot = current_rot[i, env_active_rotations]

            current_joint_vecs = torch.zeros_like(env_current_rot[..., 0:3])
            current_joint_vecs[..., 0] = 0.2
            current_joint_vecs = torch_utils.quat_rotate(
                env_current_rot, current_joint_vecs, self.w_last
            )
            current_joint_vecs += env_current_pos

            target_joint_vecs = torch.zeros_like(env_target_rot[..., 0:3])
            target_joint_vecs[..., 0] = 0.2
            target_joint_vecs = torch_utils.quat_rotate(
                env_target_rot, target_joint_vecs, self.w_last
            )
            target_joint_vecs += env_current_pos

            verts = (
                torch.cat(
                    [
                        env_current_pos,
                        current_joint_vecs,
                        env_current_pos,
                        target_joint_vecs,
                    ],
                    dim=-1,
                )
                .cpu()
                .numpy()
            )
            verts = verts.reshape([-1, 6])

            self.gym.add_lines(
                self.viewer, env_ptr, verts.shape[0], verts, rotation_cols
            )
