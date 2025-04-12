# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from typing import Optional, Dict

from isaac_utils import torch_utils, rotations
import torch
from isaac_utils.rotations import quat_mul
from isaacgym import gymapi, gymtorch
from isaac_utils.torch_utils import *
from torch import Tensor

from phys_anim.envs.masked_mimic_inversion.base_task.isaacgym import (
    MaskedMimicTaskHumanoid,
)


class MaskedMimicStrike(MaskedMimicTaskHumanoid):
    def __init__(self, config, device: torch.device, motion_lib: Optional[torch.Tensor] = None):
        super().__init__(config=config, device=device)
        self.enable_success_termination = getattr(self.config.strike_params, "enable_success_termination", False)
        if not self.headless:
            self._build_marker_state_tensors()

        self._tar_dist_min = self.config.strike_params.tar_dist_min
        self._tar_dist_max = self.config.strike_params.tar_dist_max
        self._near_dist = self.config.strike_params.near_dist
        self._near_prob = self.config.strike_params.near_prob
        self._tar_speed = self.config.strike_params.get("tar_speed", 1.0)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        self.condition_body_part = "Head"

        strike_body_names = self.config.strike_params.strike_body_names
        self._strike_body_ids = self.build_body_ids_tensor(strike_body_names)
        self._build_target_tensors()
        self._current_successes = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)

    def create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._marker_handles = [[] for _ in range(num_envs)]

        self._load_target_asset()

        super().create_envs(num_envs, spacing, num_per_row)

    def create_hand_crafted_prior(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        num_envs = len(env_ids)

        bodies_positions = self.get_body_positions()[env_ids]

        body_part = self.gym.find_asset_rigid_body_index(
            self.humanoid_asset, self.condition_body_part
        )
        head_position = bodies_positions[:, body_part, :]
        ground_below_head = self.get_ground_heights(bodies_positions[:, 0, :2])
        head_position[..., 2] -= ground_below_head.view(-1)

        tar_states = self._target_states[env_ids]

        dir_to_target = tar_states[..., :2] - head_position[..., :2]
        angle = rotations.vec_to_heading(dir_to_target).view(
            head_position.shape[0], -1
        )
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        target_direction = rotations.heading_to_quat(angle, w_last=self.w_last).view(
            head_position.shape[0], 4
        )

        distance_to_target = torch.norm(dir_to_target, dim=-1)
        close_to_target = distance_to_target < 2
        far_from_target = ~close_to_target

        single_step_mask_size = self.num_conditionable_bodies * 2
        new_mask = torch.zeros(
            num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.device,
        )
        pelvis_body_index = self.config.masked_mimic_conditionable_bodies.index(
            "Pelvis"
        )
        head_body_index = self.config.masked_mimic_conditionable_bodies.index(
            "Head"
        )
        self.motion_times[:] = 0
        self.target_pose_time[:] = (
                self.motion_times[:] + 1.0
        )
        new_mask[:, pelvis_body_index, :] = True  # translation + rotation
        new_mask[:, head_body_index, :] = True  # translation + rotation
        new_mask[:, -1, :] = True  # velocity + rotation
        new_mask = (
            new_mask.view(num_envs, 1, single_step_mask_size)
            .expand(-1, self.config.masked_mimic_obs.num_future_steps, -1)
            .reshape(num_envs, -1)
        )

        # Set the mask, needed here since obs = [maskedmimic_obs * masks]
        self.masked_mimic_target_bodies_masks[env_ids, :] = new_mask
        # Define the mask for the "far away pose"
        self.target_pose_obs_mask[env_ids[far_from_target]] = True
        self.target_pose_obs_mask[env_ids[close_to_target]] = False
        # Set the "far away pose" constraints
        self.target_pose_joints[env_ids] = False
        self.target_pose_joints[env_ids, pelvis_body_index * 2 + 1] = True
        self.target_pose_joints[env_ids, head_body_index * 2 + 1] = True

        # Compute MM obs
        self.masked_mimic_target_poses[env_ids] = (
            self.build_sparse_target_object_poses_masked_with_time(
                env_ids,
                self.config.masked_mimic_obs.num_future_steps,
                target_direction,
                tar_states[..., :2]
            )
        )

        # Set the transformer masks
        self.masked_mimic_target_poses_masks[env_ids, :] = False
        self.masked_mimic_target_poses_masks[env_ids[far_from_target], -2:] = True
        self.motion_text_embeddings_mask[env_ids] = False

        # Envs that succeeded --> remove constraint.
        tar_pos = self._target_states[env_ids, :3]
        tar_rot = self._target_states[env_ids, 3:7]
        up = torch.zeros_like(tar_pos)
        up[..., -1] = 1
        tar_up = quat_rotate(tar_rot, up, w_last=self.w_last)
        tar_rot_err = torch.sum(up * tar_up, dim=-1)
        succ = tar_rot_err < 0.2
        self.masked_mimic_target_poses_masks[env_ids[succ], :] = False

        # self.motion_text_embeddings_mask[env_ids[close_to_target]] = True

        # self.motion_text_embeddings_mask[env_ids] = False
        # self.motion_text_embeddings_mask[env_ids[close_to_target]] = True
        # self.motion_text_embeddings[:] = self._text_embedding

    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)

    def _load_target_asset(self):
        asset_root = "phys_anim/data/assets/urdf/"
        asset_file = "strike_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group,
                                              col_filter, segmentation_id)
        self._target_handles.append(target_handle)

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self.root_states.view(self.num_envs, num_actors, self.root_states.shape[-1])[..., 1, :]

        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        bodies_per_env = self.rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)
        self._reset_strike_target(env_ids)

    def reset_task(self, env_ids=None):
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = self._last_length[env_ids] > 0
            average_distances = (self._current_accumulated_errors[env_ids][active_envs] /
                                 self._last_length[env_ids][active_envs])
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_successes[env_ids][active_envs] == 0).cpu().tolist()
            )
            # for the last episode, we need to accumulate the errors
            self.accumulate_errors()

            self._current_successes[env_ids] = 0
            self._reset_strike_target(env_ids)
        super().reset_task(env_ids)

    def _reset_strike_target(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_states = self.get_humanoid_root_states()

        n = len(env_ids)

        init_near = torch.rand([n], dtype=self._target_states.dtype,
                               device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                   device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_dist_min

        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + root_states[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + root_states[env_ids, 1]
        self._target_states[env_ids, 2] = 0.9

        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis, w_last=self.w_last)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0

    def reset_env_tensors(self, env_ids):
        super().reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self.get_humanoid_root_states()[..., 0:3]

    def compute_task_obs(self, env_ids=None):
        super().compute_task_obs(env_ids)
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        root_states = self.get_humanoid_root_states()[env_ids]
        tar_states = self._target_states[env_ids]

        obs = compute_strike_observations(root_states, tar_states)
        self.inversion_obs[env_ids] = torch.cat([obs, self.current_pose_obs], dim=-1)

    def compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        char_root_state = self.humanoid_root_states
        strike_body_vel = self.rigid_body_vel[..., self._strike_body_ids[0], :]

        self.rew_buf[:], output_dict = compute_strike_reward(tar_pos,
                                                             tar_rot,
                                                             char_root_state,
                                                             self._prev_root_pos,
                                                             self.dt,
                                                             self._tar_speed,
                                                             self.w_last)

        if (
                self.config.num_envs == 1
                and self.config.get("log_output", False)
                and self.progress_buf % 3 == 0
        ):
            self.print_results(output_dict)

        self.log_dict.update(output_dict)
        # # need these at the end of every compute_reward function
        self.compute_failures_and_distances()
        self.accumulate_errors()

    def compute_failures_and_distances(self):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        up = torch.zeros_like(tar_pos)
        up[..., -1] = 1
        tar_up = quat_rotate(tar_rot, up, w_last=self.w_last)
        tar_rot_err = torch.sum(up * tar_up, dim=-1)

        distance_to_target = torch.norm(self.humanoid_root_states[..., 0:3] - tar_pos, dim=-1)

        self._current_accumulated_errors[:] += distance_to_target
        self._current_successes[:] = tar_rot_err < 0.2
        self._last_length[:] = self.progress_buf[:]

    def compute_reset(self):
        bodies_positions = self.get_body_positions()

        bodies_positions[..., 2] -= (
            torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        )

        termination_heights = self.termination_heights + self.get_ground_heights(
            bodies_positions[:, self.head_body_id, :2])
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(tar_pos, tar_rot, self.reset_buf,
                                                                          self.progress_buf, self.contact_forces,
                                                                          self.non_termination_contact_body_ids,
                                                                          self.rigid_body_pos, self._tar_contact_forces,
                                                                          self._strike_body_ids,
                                                                          self.config.max_episode_length,
                                                                          self.config.enable_height_termination,
                                                                          termination_heights,
                                                                          self.enable_success_termination)

    def build_sparse_target_object_poses(
            self, env_ids, raw_future_times, target_directions, target_positions
    ):
        """
        This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        num_envs = len(env_ids)
        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids[env_ids].unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos, flat_target_rot, flat_target_vel = (
            ref_state.rb_pos,
            ref_state.rb_rot,
            ref_state.rb_vel,
        )

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = current_state.body_pos[env_ids], current_state.body_rot[env_ids]
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            num_envs, 1, 1
        )
        # cur_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(self.num_envs, 1, 2)

        # override to set the target root parameters
        reshaped_target_pos = flat_target_pos.reshape(
            num_envs, num_future_steps, -1, 3
        )
        reshaped_target_rot = flat_target_rot.reshape(
            num_envs, num_future_steps, -1, 4
        )

        # turned_envs = ~turning_envs
        cur_pos = cur_gt[:, 0, :2]
        tar_dir = (target_positions - cur_pos) / torch.norm(target_positions - cur_pos, dim=-1, keepdim=True)

        reshaped_target_pos[:, :, :, :2] = cur_gt[:, 0, :2].unsqueeze(1).unsqueeze(1).clone()
        for frame_idx in range(num_future_steps):
            reshaped_target_pos[:, frame_idx, :, :2] += (
                    tar_dir.view(num_envs, 1, 2)
                    * self._tar_speed
                    * (raw_future_times[:, frame_idx] - self.motion_times[env_ids]).view(num_envs, 1, 1)
            )

        reshaped_target_pos[:, :, 0, -1] = 0.88  # standing up
        reshaped_target_pos[:, :, 1:, -1] = 1.5  # standing up

        # angle = rotations.vec_to_heading(self._tar_facing_dir)
        reshaped_target_rot[:] = target_directions.view(num_envs, 1, 1, 4)

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)
        flat_target_rot = reshaped_target_rot.reshape(flat_target_rot.shape)

        non_flat_target_vel = flat_target_vel.reshape(
            num_envs, num_future_steps, -1, 3
        )

        non_flat_target_vel[:, :, 0, :2] = (
                tar_dir[:] * self._tar_speed
        ).view(tar_dir.shape[0], 1, 2)
        flat_target_vel = non_flat_target_vel.reshape(flat_target_vel.shape)
        # override to set the target root parameters

        expanded_body_pos = cur_gt.unsqueeze(1).expand(
            num_envs, num_future_steps, *cur_gt.shape[1:]
        )
        expanded_body_rot = cur_gr.unsqueeze(1).expand(
            num_envs, num_future_steps, *cur_gr.shape[1:]
        )

        flat_cur_pos = expanded_body_pos.reshape(flat_target_pos.shape)
        flat_cur_rot = expanded_body_rot.reshape(flat_target_rot.shape)

        root_pos = flat_cur_pos[:, 0, :]
        root_rot = flat_cur_rot[:, 0, :]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, flat_cur_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(
            heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
            heading_rot_expand.shape[2],
        )

        root_pos_expand = root_pos.unsqueeze(-2)

        """target"""
        # target body pos   [N, 3xB]
        target_rel_body_pos = flat_target_pos - flat_cur_pos
        flat_target_rel_body_pos = target_rel_body_pos.reshape(
            target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
            target_rel_body_pos.shape[2],
        )
        flat_target_rel_body_pos = torch_utils.quat_rotate(
            flat_heading_rot, flat_target_rel_body_pos, self.w_last
        )

        # target body pos   [N, 3xB]
        flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
            flat_target_pos.shape[0] * flat_target_pos.shape[1],
            flat_target_pos.shape[2],
        )
        flat_target_body_pos = torch_utils.quat_rotate(
            flat_heading_rot, flat_target_body_pos, self.w_last
        )

        # target body rot   [N, 6xB]
        target_rel_body_rot = rotations.quat_mul(
            rotations.quat_conjugate(flat_cur_rot, self.w_last),
            flat_target_rot,
            self.w_last,
        )
        target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(
            target_rel_body_rot.view(-1, 4), self.w_last
        ).view(target_rel_body_rot.shape[0], -1)

        # target body rot   [N, 6xB]
        target_body_rot = rotations.quat_mul(
            heading_rot_expand, flat_target_rot, self.w_last
        )
        target_body_rot_obs = torch_utils.quat_to_tan_norm(
            target_body_rot.view(-1, 4), self.w_last
        ).view(target_rel_body_rot.shape[0], -1)

        padded_flat_target_rel_body_pos = torch.nn.functional.pad(
            flat_target_rel_body_pos, [0, 3], "constant", 0
        )
        sub_sampled_target_rel_body_pos = padded_flat_target_rel_body_pos.reshape(
            num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        padded_flat_target_body_pos = torch.nn.functional.pad(
            flat_target_body_pos, [0, 3], "constant", 0
        )
        sub_sampled_target_body_pos = padded_flat_target_body_pos.reshape(
            num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        sub_sampled_target_rel_body_rot_obs = target_rel_body_rot_obs.reshape(
            num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]
        sub_sampled_target_body_rot_obs = target_body_rot_obs.reshape(
            num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        # Heading
        target_heading_rot = torch_utils.calc_heading_quat(
            flat_target_rot[:, 0, :], self.w_last
        )
        target_rel_heading_rot = torch_utils.quat_to_tan_norm(
            rotations.quat_mul(
                rotations.quat_conjugate(heading_rot_expand[:, 0, :], self.w_last),
                target_heading_rot,
                self.w_last,
            ).view(-1, 4),
            self.w_last,
        ).reshape(num_envs, num_future_steps, 1, 6)

        # Velocity
        target_root_vel = flat_target_vel[:, 0, :]
        target_root_vel[..., -1] = 0  # ignore vertical speed
        target_rel_vel = rotations.quat_rotate(
            heading_rot, target_root_vel, self.w_last
        ).reshape(-1, 3)
        padded_target_rel_vel = torch.nn.functional.pad(
            target_rel_vel, [0, 3], "constant", 0
        )
        padded_target_rel_vel = padded_target_rel_vel.reshape(
            num_envs, num_future_steps, 1, 6
        )

        heading_and_velocity = torch.cat(
            [
                target_rel_heading_rot,
                target_rel_heading_rot,
                padded_target_rel_vel,
                padded_target_rel_vel,
            ],
            dim=-1,
        )

        # In masked_mimic allow easy re-shape to [batch, time, joint, type (transform/rotate), features]
        obs = torch.cat(
            (
                sub_sampled_target_rel_body_pos,
                sub_sampled_target_body_pos,
                sub_sampled_target_rel_body_rot_obs,
                sub_sampled_target_body_rot_obs,
            ),
            dim=-1,
        )  # [batch, timesteps, joints, 24]
        obs = torch.cat((obs, heading_and_velocity), dim=-2).view(num_envs, -1)

        return obs

    def build_sparse_target_object_poses_masked_with_time(
            self, env_ids, num_future_steps, target_directions, target_position
    ):
        num_envs = len(env_ids)
        time_offsets = (
                torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
                * self.dt
        )

        near_future_times = self.motion_times[env_ids].unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time[env_ids].view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_object_poses(
            env_ids, all_future_times, target_directions, target_position
        ).view(
            num_envs,
            num_future_steps + 1,
            self.masked_mimic_conditionable_bodies_ids.shape[0] + 1,
            2,
            12,
        )

        near_mask = self.masked_mimic_target_bodies_masks[env_ids].view(
            num_envs, num_future_steps, self.num_conditionable_bodies, 2, 1
        )
        far_mask = self.target_pose_joints[env_ids].view(num_envs, 1, -1, 2, 1)
        mask = torch.cat([near_mask, far_mask], dim=1)

        masked_obs = obs * mask

        masked_obs_with_joints = torch.cat((masked_obs, mask), dim=-1).view(
            num_envs, num_future_steps + 1, -1
        )

        times = all_future_times.view(-1).view(
            num_envs, num_future_steps + 1, 1
        ) - self.motion_times[env_ids].view(num_envs, 1, 1)
        ones_vec = torch.ones(
            num_envs, num_future_steps + 1, 1, device=self.device
        )
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(num_envs, -1)


    def draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self.humanoid_root_states[..., 0:3]
        ends = self._target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        if not self.config.get("clean_video", False):
            for i, env_ptr in enumerate(self.envs):
                curr_verts = verts[i]
                curr_verts = curr_verts.reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_strike_observations(root_states, tar_states, w_last=True):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last=w_last)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos, w_last=w_last)
    local_tar_vel = quat_rotate(heading_rot, tar_vel, w_last=w_last)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel, w_last=w_last)

    local_tar_rot = quat_mul(heading_rot, tar_rot, w_last=w_last)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot, w_last=w_last)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
    return obs


@torch.jit.script
def compute_strike_reward(tar_pos, tar_rot, root_state, prev_root_pos, dt, tar_speed, w_last=True):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, bool) -> Tuple[Tensor, Dict[str, Tensor]]
    # tar_speed = 1.0
    # tar_speed = 2.0
    vel_err_scale = 4.0

    tar_rot_w = 0.6
    vel_reward_w = 0.4

    up = torch.zeros_like(tar_pos)
    up[..., -1] = 1
    tar_up = quat_rotate(tar_rot, up, w_last=w_last)
    tar_rot_err = torch.sum(up * tar_up, dim=-1)
    tar_rot_r = torch.clamp_min(1.0 - tar_rot_err, 0.0)

    root_pos = root_state[..., 0:3]
    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err_rel = tar_vel_err / tar_speed  # it's 1.0 always
    tar_vel_err_rel = torch.clamp_min(tar_vel_err_rel, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err_rel * tar_vel_err_rel))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    reward = tar_rot_w * tar_rot_r + vel_reward_w * vel_reward

    succ = tar_rot_err < 0.2
    reward = torch.where(succ, torch.ones_like(reward), reward)

    output_dict = {
        "tar_rot_err": tar_rot_err,
        "tar_rot_r": tar_rot_r,
        "vel_reward": vel_reward,
    }
    return reward, output_dict


@torch.jit.script
def compute_humanoid_reset(tar_pos, tar_rot, reset_buf, progress_buf, contact_buf, non_termination_contact_body_ids,
                           rigid_body_pos, tar_contact_forces, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, enable_success_termination):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, bool) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0

    terminated = torch.zeros_like(reset_buf)
    success = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, non_termination_contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, non_termination_contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        # strike_body_force = contact_buf[:, strike_body_id, :]
        # strike_body_has_contact = torch.any(torch.abs(strike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)

        has_failed = torch.logical_or(has_fallen, tar_fail)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

        # Define success condition: target falls
        up = torch.zeros_like(tar_pos)
        up[..., -1] = 1
        tar_up = quat_rotate(tar_rot, up, w_last=True)
        tar_rot_err = torch.sum(up * tar_up, dim=-1)
        success = tar_rot_err < 0.2
        success *= (progress_buf > 1)

        if not enable_success_termination:
            success = torch.zeros_like(success)

    combined_reset = success | (progress_buf >= max_episode_length - 1) | terminated
    reset = torch.where(combined_reset.to(torch.bool), torch.ones_like(reset_buf), reset_buf)

    return reset, terminated
