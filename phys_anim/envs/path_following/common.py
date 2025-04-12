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

from isaac_utils import torch_utils, rotations

import torch
from torch import Tensor

from phys_anim.envs.env_utils.path_generator import PathGenerator
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from phys_anim.envs.path_following.isaacgym import PathFollowingHumanoid
else:
    PathFollowingHumanoid = object


class BasePathFollowing(PathFollowingHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        self._num_traj_samples = self.config.path_follower_params.num_traj_samples
        self._traj_sample_timestep = (
            self.config.path_follower_params.traj_sample_timestep
        )

        self.path_obs = torch.zeros(
            (
                self.config.num_envs,
                self.config.path_follower_params.path_obs_size,
            ),
            device=device,
            dtype=torch.float,
        )

        self._fail_dist = 4.0
        self._fail_height_dist = 0.5

        self.build_path_generator()
        self.reset_path_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )

    ###############################################################
    # Handle resets
    ###############################################################
    def reset_task(self, env_ids):
        super().reset_task(env_ids)
        self.reset_path_ids = env_ids

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_task_obs(self, env_ids=None):
        super().compute_task_obs(env_ids)

        bodies_positions = self.get_body_positions()

        if env_ids is None:
            root_states = self.get_humanoid_root_states()
            head_position = bodies_positions[:, self.head_body_id, :]
            ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
            head_position = bodies_positions[env_ids, self.head_body_id, :]
            ground_below_head = torch.min(bodies_positions[env_ids], dim=1).values[
                ..., 2
            ]

        if self.reset_path_ids is not None and len(self.reset_path_ids) > 0:
            reset_head_position = bodies_positions[
                self.reset_path_ids, self.head_body_id, :
            ]
            flat_reset_head_position = reset_head_position.view(-1, 3)
            ground_below_reset_head = self.get_ground_heights(
                bodies_positions[:, self.head_body_id, :2]
            )[self.reset_path_ids]
            flat_reset_head_position[..., 2] -= ground_below_reset_head.view(-1)
            self.path_generator.reset(self.reset_path_ids, flat_reset_head_position)

            self.reset_path_ids = None

        traj_samples = self.fetch_path_samples(env_ids)

        flat_head_position = head_position.view(-1, 3)
        flat_head_position[..., 2] -= ground_below_head.view(-1)

        obs = compute_path_observations(
            root_states,
            flat_head_position,
            traj_samples,
            self.w_last,
            self.config.path_follower_params.height_conditioned,
        )

        self.path_obs[env_ids] = obs

    def compute_reward(self, actions):
        bodies_positions = self.get_body_positions()
        head_position = bodies_positions[:, self.head_body_id, :]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position[..., 2] -= ground_below_head.view(-1)

        self.rew_buf[:] = compute_path_reward(
            head_position, tar_pos, self.config.path_follower_params.height_conditioned
        )

    def compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        bodies_positions = self.get_body_positions()
        bodies_contact_buf = self.get_bodies_contact_buf()

        bodies_positions[..., 2] -= (
            torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        )

        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.non_termination_contact_body_ids,
            bodies_positions,
            tar_pos,
            self.config.max_episode_length,
            self._fail_dist,
            self._fail_height_dist,
            self.config.enable_height_termination,
            self.config.path_follower_params.enable_path_termination,
            self.config.path_follower_params.height_conditioned,
            self.termination_heights
            + self.get_ground_heights(bodies_positions[:, self.head_body_id, :2]),
            self.head_body_id,
        )

    ###############################################################
    # Helpers
    ###############################################################
    def build_path_generator(self):
        episode_dur = self.config.max_episode_length * self.dt
        self.path_generator = PathGenerator(
            self.config.path_follower_params.path_generator,
            self.device,
            self.num_envs,
            episode_dur,
            self.config.path_follower_params.height_conditioned,
        )

    def fetch_path_samples(self, env_ids=None):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(
            self._num_traj_samples, device=self.device, dtype=torch.float
        )
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self._num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )

        return traj_samples


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_path_observations(
    root_states: Tensor,
    head_states: Tensor,
    traj_samples: Tensor,
    w_last: bool,
    height_conditioned: bool,
) -> Tensor:
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )

    traj_samples_delta = traj_samples - head_states.unsqueeze(-2)

    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )

    local_traj_pos = rotations.quat_rotate(
        heading_rot_exp, traj_samples_delta_flat, w_last
    )
    if not height_conditioned:
        local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs


@torch.jit.script
def compute_path_reward(head_pos, tar_pos, height_conditioned):
    # type: (Tensor, Tensor, bool) -> Tensor
    pos_err_scale = 2.0
    height_err_scale = 10.0

    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        reward = (pos_reward + height_reward) * 0.5
    else:
        reward = pos_reward

    return reward


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    non_termination_contact_body_ids,
    rigid_body_pos,
    tar_pos,
    max_episode_length,
    fail_dist,
    fail_height_dist,
    enable_early_termination,
    enable_path_termination,
    enable_height_termination,
    termination_heights,
    head_body_id,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, bool, bool, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

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
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
    else:
        has_fallen = progress_buf < -1

    if enable_path_termination:
        head_pos = rigid_body_pos[..., head_body_id, :]
        tar_delta = tar_pos - head_pos
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_overall_fail = tar_dist_sq > fail_dist * fail_dist

        if enable_height_termination:
            tar_height = tar_pos[..., 2]
            height_delta = tar_height - head_pos[..., 2]
            tar_head_dist_sq = height_delta * height_delta
            tar_height_fail = tar_head_dist_sq > fail_height_dist * fail_height_dist
            tar_height_fail *= progress_buf > 20

            tar_fail = torch.logical_or(tar_overall_fail, tar_height_fail)
        else:
            tar_fail = tar_overall_fail
    else:
        tar_fail = progress_buf < -1

    has_failed = torch.logical_or(has_fallen, tar_fail)

    terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated
