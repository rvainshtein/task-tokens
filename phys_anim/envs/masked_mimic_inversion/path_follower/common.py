from isaac_utils import torch_utils, rotations

import torch
from torch import Tensor

from typing import Optional, Tuple, Dict

from phys_anim.envs.env_utils.path_generator import PathGenerator
from phys_anim.utils.motion_lib import MotionLib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic_inversion.path_follower.isaacgym import (
        MaskedMimicPathFollowingHumanoid,
    )
else:
    MaskedMimicPathFollowingHumanoid = object


class BaseMaskedMimicPathFollowing(MaskedMimicPathFollowingHumanoid):  # type: ignore[misc]
    def __init__(
            self, config, device: torch.device, motion_lib: Optional[MotionLib] = None
    ):
        super().__init__(config=config, device=device, motion_lib=motion_lib)
        self.path_obs = torch.zeros(
            self.config.num_envs,
            self.config.path_follower_params.path_obs_size,
            device=device,
            dtype=torch.float,
        )

        self._num_traj_samples = self.config.path_follower_params.num_traj_samples
        self._traj_sample_timestep = (
            self.config.path_follower_params.traj_sample_timestep
        )

        self.condition_body_part = "Head"

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
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = self._current_accumulated_errors[env_ids] > 0
            average_distances = (
                    self._current_accumulated_errors[env_ids][active_envs]
                    / self._last_length[env_ids][active_envs]
            )
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_failures[env_ids][active_envs] > 0).cpu().tolist()
            )
            self._current_failures[env_ids] = 0
        super().reset_task(env_ids)
        self.reset_path_ids = env_ids

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_task_obs(self, env_ids=None):
        super().compute_task_obs(env_ids)

        bodies_positions = self.get_body_positions()

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        root_states = self.get_humanoid_root_states()[env_ids]
        head_position = bodies_positions[env_ids, self.head_body_id, :]
        ground_below_head = torch.min(bodies_positions[env_ids], dim=1).values[..., 2]

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

        traj_samples = self.fetch_path_samples(env_ids, time_offset=0)[0]

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
        self.inversion_obs[env_ids] = torch.cat(
            [self.path_obs[env_ids], self.current_pose_obs], dim=-1
        )

    def compute_reward(self, actions):
        bodies_positions = self.get_body_positions()
        head_position = bodies_positions[:, self.head_body_id, :]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position[..., 2] -= ground_below_head.view(-1)

        self.rew_buf[:], output_dict = compute_path_reward(
            head_position, tar_pos, self.config.path_follower_params.height_conditioned
        )

        if (
                self.config.num_envs == 1
                and self.config.get("log_output", False)
                and self.progress_buf % 3 == 0
        ):
            self.print_results(output_dict)

        self.log_dict.update(output_dict)
        # need these at the end of every compute_reward function
        self.compute_failures_and_distances()
        self.accumulate_errors()

    def compute_failures_and_distances(self):
        body_part = self.gym.find_asset_rigid_body_index(
            self.humanoid_asset, self.condition_body_part
        )
        current_state = self.get_bodies_state()
        cur_gt = current_state.body_pos

        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time).clone()

        distance_to_target = torch.norm(
            cur_gt[:, body_part, :3] - tar_pos[:, :3], dim=-1
        ).view(self.num_envs)

        warmup_passed = self.progress_buf > 10  # 10 frames

        self._current_accumulated_errors[warmup_passed] += distance_to_target[
            warmup_passed
        ]
        self._current_failures[warmup_passed] += distance_to_target[warmup_passed] > 2.0
        self._last_length[warmup_passed] = self.progress_buf[warmup_passed]

        self._current_accumulated_errors[~warmup_passed] = 0
        self._current_failures[~warmup_passed] = 0

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

        if self.reset_path_ids is not None and len(self.reset_path_ids) > 0:
            self.path_generator.reset(
                self.reset_path_ids, head_position[self.reset_path_ids]
            )

            assert (
                    self.progress_buf[self.reset_path_ids] <= 1
            ).all(), (
                f"Progress should be reset {self.progress_buf[self.reset_path_ids]}"
            )

            self.reset_path_ids = None
        traj_samples, traj_samples_p1 = self.fetch_path_samples(env_ids)
        dir_p_to_p_1 = traj_samples_p1[..., :2] - traj_samples[..., :2]
        dir_p_to_p_1_flat = dir_p_to_p_1.view(-1, 2)
        angle = rotations.vec_to_heading(dir_p_to_p_1_flat).view(
            dir_p_to_p_1_flat.shape[0], -1
        )
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        direction = rotations.heading_to_quat(angle, w_last=self.w_last).view(
            dir_p_to_p_1.shape[0], -1, 4
        )

        body_index = self.config.masked_mimic_conditionable_bodies.index(
            self.condition_body_part
        )
        single_step_mask_size = self.num_conditionable_bodies * 2
        new_mask = torch.zeros(
            num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.device,
        )
        new_mask[:, body_index, 0] = True
        # new_mask[:, -1, :] = True  # heading & speed
        new_mask = (
            new_mask.view(num_envs, 1, single_step_mask_size)
            .expand(-1, self.config.masked_mimic_obs.num_future_steps, -1)
            .reshape(num_envs, -1)
        )
        # new_mask = new_mask.view(self.num_envs, 1, single_step_mask_size).expand(-1, self.config.num_future_steps, -1).reshape(self.num_envs, -1)
        self.masked_mimic_target_bodies_masks[env_ids] = new_mask
        # self.masked_mimic_target_bodies_masks[:] = new_mask

        self.target_pose_joints[env_ids] = False
        self.target_pose_joints[env_ids, body_index * 2] = True
        self.target_pose_joints[env_ids, body_index * 2 + 1] = True
        # self.target_pose_joints[:, -2:] = True  # heading & speed
        self.target_pose_time[env_ids] = (
                self.motion_times[env_ids] + self._traj_sample_timestep
        )

        target_poses = self.build_sparse_target_path_poses_masked_with_time(
            traj_samples, direction, env_ids
        )
        self.masked_mimic_target_poses[env_ids] = target_poses

        # self.masked_mimic_target_poses_masks[env_ids] = True
        self.masked_mimic_target_poses_masks[:, 5] = True
        # # self.masked_mimic_target_poses_masks[:, 1] = True
        # self.masked_mimic_target_poses_masks[:, 2] = True
        self.masked_mimic_target_poses_masks[:, -1] = True

        too_far = (
                torch.norm(traj_samples[:, 0, :2] - bodies_positions[:, 0, :2], dim=-1)
                > 0.4
        )
        self.masked_mimic_target_poses_masks[env_ids[too_far], :-1] = False

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
            self.config.path_follower_params.path_generator.height_conditioned,
        )

    def fetch_path_samples(self, env_ids=None, time_offset=10):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(
            self._num_traj_samples, device=self.device, dtype=torch.float
        )
        if time_offset == 0:
            timesteps = timesteps * self._traj_sample_timestep
        else:
            timesteps = (timesteps + 1) * self.dt
            timesteps[-1] += self._traj_sample_timestep
        timesteps_p1 = timesteps + self.dt
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps
        traj_timesteps_p1 = timestep_beg.unsqueeze(-1) + timesteps_p1

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        ).clone()
        traj_samples_p1_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps_p1.flatten()
        ).clone()
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self._num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )
        traj_samples_p1 = torch.reshape(
            traj_samples_p1_flat,
            shape=(
                env_ids.shape[0],
                self._num_traj_samples,
                traj_samples_p1_flat.shape[-1],
            ),
        )

        return traj_samples, traj_samples_p1

    def build_sparse_target_path_poses(
            self, raw_future_times, target_root_pos, target_root_rot, env_ids
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
        cur_gt, cur_gr = (
            current_state.body_pos[env_ids],
            current_state.body_rot[env_ids],
        )
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            num_envs, 1, 1
        )

        # override to set the target root parameters
        reshaped_target_pos = flat_target_pos.reshape(num_envs, num_future_steps, -1, 3)
        reshaped_target_rot = flat_target_rot.reshape(num_envs, num_future_steps, -1, 4)

        body_part = self.gym.find_asset_rigid_body_index(
            self.humanoid_asset, self.condition_body_part
        )

        reshaped_target_pos[:, :, body_part, :3] = target_root_pos[..., :3]
        reshaped_target_rot[:, :, body_part] = target_root_rot[:]

        # reshaped_target_pos[:, :, body_part, -1] = 0.92  # standing up

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)
        flat_target_rot = reshaped_target_rot.reshape(flat_target_rot.shape)
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
                heading_rot_expand[:, 0, :], target_heading_rot, self.w_last
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

    def build_sparse_target_path_poses_masked_with_time(
            self, target_root_pos: Tensor, target_root_rot: Tensor, env_ids
    ):
        num_envs = len(env_ids)
        num_future_steps = target_root_pos.shape[1] - 1
        time_offsets = (
                torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
                * self.dt
        )

        near_future_times = self.motion_times[env_ids].unsqueeze(
            -1
        ) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time[env_ids].view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_path_poses(
            all_future_times, target_root_pos, target_root_rot, env_ids
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
        ones_vec = torch.ones(num_envs, num_future_steps + 1, 1, device=self.device)
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(num_envs, -1)


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
    # type: (Tensor, Tensor, bool) -> Tuple[Tensor, Dict[str, Tensor]]
    pos_err_scale = 2.0
    height_err_scale = 10.0

    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        # Compute weighted harmonic mean
        pos_weight = 0.5
        reward = 2 * pos_weight * pos_reward * height_reward / (
                    pos_weight * pos_reward + (1 - pos_weight) * height_reward + 1e-6)

    else:
        reward = pos_reward
    output_dict = dict(pos_reward=pos_reward,
                       height_reward=height_reward,
                       tar_pos=tar_pos[..., 0:2],
                       head_pos=head_pos[..., 0:2],
                       height_tar=tar_pos[..., 2],
                       height_head=head_pos[..., 2])
    return reward, output_dict


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
