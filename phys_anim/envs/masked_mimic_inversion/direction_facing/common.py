import numpy as np
import torch
from isaac_utils import torch_utils, rotations
from isaac_utils.rotations import quat_rotate
from isaac_utils.torch_utils import calc_heading_quat
from typing import TYPE_CHECKING, Dict, Tuple

from phys_anim.envs.humanoid.humanoid_utils import quat_diff_norm
from torch import Tensor

from phys_anim.envs.masked_mimic_inversion.steering.common import compute_heading_reward

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic_inversion.direction_facing.isaacgym import (
        MaskedMimicDirectionFacingHumanoid,
    )

else:
    MaskedMimicDirectionFacingHumanoid = object


class MaskedMimicBaseDirectionFacing(MaskedMimicDirectionFacingHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        self._tar_facing_dir_theta = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._tar_facing_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_facing_dir[..., 0] = 1.0

        self._heading_turn_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )

        self.facing_obs = torch.zeros(
            (self.num_envs, 2), device=device, dtype=torch.float
        )

    def compute_task_obs(self, env_ids=None):
        super().compute_task_obs(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_states = self.get_humanoid_root_states()[env_ids]

        facing_obs = compute_facing_observations(
            root_states, self._tar_facing_dir[env_ids], self.w_last
        )
        self.facing_obs[env_ids] = facing_obs
        self.inversion_obs[env_ids] = torch.cat(
            [self.direction_obs[env_ids], self.facing_obs[env_ids], self.current_pose_obs], dim=-1)

    def compute_reward(self, actions):
        root_states = self.get_humanoid_root_states()
        root_pos = root_states[..., :3]
        root_rot = root_states[:, 3:7]
        self.rew_buf[:], output_dict = compute_facing_reward(
            root_pos,
            self._prev_root_pos,
            root_rot,
            self._tar_dir,
            self._tar_speed,
            self._tar_facing_dir,
            self.dt,
            self.w_last,
        )

        # print the target speed of the env and the speed actually achieved in that direction

        if (
                self.config.num_envs == 1
                # and self.config.steering_params.log_speed
                and self.config.get("log_output", False)
                and self.progress_buf % 3 == 0
        ):
            results_output = {
                "Speed": f'{output_dict["tar_dir_speed"].item():.3f} / {self._tar_speed.item():.3f}',
                "Error": f'{output_dict["tar_vel_err"].item():.3f}',
                "Tangent Error": f'{output_dict["tangent_vel_err"].item():.3f}',
            }
            self.print_results(results_output)

        self.log_dict.update(output_dict)
        # need these at the end of every compute_reward function
        self.compute_failures_and_distances()
        self.accumulate_errors()
        # It's important that it's here after calculation of failures and distances
        self._prev_root_pos[:] = root_pos


    def compute_failures_and_distances(self):
        current_state = self.get_bodies_state()
        body_pos, body_rot = (
            current_state.body_pos,
            current_state.body_rot,
        )
        turning_envs = self._heading_turn_steps > self.progress_buf
        turned_envs = ~turning_envs

        delta_root_pos = self.get_humanoid_root_states()[..., :3] - self._prev_root_pos[:]
        root_vel = delta_root_pos / self.dt
        tar_dir_speed = torch.sum(self._tar_dir * root_vel[..., :2], dim=-1)

        tar_dir_vel = tar_dir_speed.unsqueeze(-1) * self._tar_dir[:]
        tangent_vel = root_vel[..., :2] - tar_dir_vel

        tangent_vel_error = torch.norm(tangent_vel, dim=-1)

        tar_vel_err = self._tar_speed[:] - tar_dir_speed
        tar_vel_err_rel = torch.where(self._tar_speed[:] > 1e-4, tar_vel_err / self._tar_speed[:], tar_vel_err)

        # Turn 3d rotation to flat heading quaternion
        facing_quat = torch_utils.calc_heading_quat(body_rot[:, 0], w_last=self.w_last)
        # Turn 2 vector to quaternion
        angle = rotations.vec_to_heading(self._tar_facing_dir)
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        tar_facing_quat = rotations.heading_to_quat(angle, w_last=self.w_last)
        # Compute angle error
        facing_err = quat_diff_norm(facing_quat, tar_facing_quat, self.w_last)
        facing_err_degrees = facing_err * 180 / torch.pi



        self._current_accumulated_errors[turned_envs] += tangent_vel_error[turned_envs]
        self._current_failures[turned_envs] += torch.logical_or(torch.abs(facing_err_degrees[turned_envs]) > 45,
                                                                torch.abs(tar_vel_err_rel[turned_envs]) > 0.2)
        self._current_failures[turning_envs] = 0
        self._current_accumulated_errors[turning_envs] = 0
        self._last_length[:] = self.progress_buf[:]

    def reset_heading_task(self, env_ids):
        super().reset_heading_task(env_ids)
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = (self._current_accumulated_errors[env_ids] > 0) & (
                    (self._last_length[env_ids] - self._heading_turn_steps[env_ids]) > 0
            )
            average_distances = self._current_accumulated_errors[env_ids][
                                    active_envs
                                ] / (
                                        self._last_length[env_ids][active_envs]
                                        - self._heading_turn_steps[env_ids][active_envs]
                                )
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_failures[env_ids][active_envs] > 0).cpu().tolist()
            )
            # for the last episode, we need to accumulate the errors
            self.accumulate_errors()

            self._current_failures[env_ids] = 0
        n = len(env_ids)
        if np.random.binomial(1, self._random_heading_probability):
            face_dir_theta = 2 * torch.pi * torch.rand(n, device=self.device) - torch.pi
        else:
            dir_delta_theta = (
                    2 * self._standard_heading_change * torch.rand(n, device=self.device)
                    - self._standard_heading_change
            )
            # map tar_dir_theta back to [0, 2pi], add delta, project back into [0, 2pi] and then shift.
            face_dir_theta = (
                                     dir_delta_theta + self._tar_facing_dir_theta[env_ids] + np.pi
                             ) % (2 * np.pi) - np.pi

        face_tar_dir = torch.stack([torch.cos(face_dir_theta), torch.sin(face_dir_theta)], dim=-1)
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._tar_facing_dir_theta[env_ids] = face_dir_theta

        self._heading_turn_steps[env_ids] = (80 * 1 + self.progress_buf[env_ids])  # Allow 15 frames (0.5sec) to turn.

    def create_hand_crafted_prior(self, env_ids):
        turning_envs = self.progress_buf < 0
        turned_envs = ~turning_envs
        # -10 just to make it reach the orientation slightly before we start measuring.
        time_left_to_turn = (self._heading_turn_steps - self.progress_buf - 10).clamp(2)
        head_body_index = self.config.masked_mimic_conditionable_bodies.index("Head")
        pelvis_body_index = self.config.masked_mimic_conditionable_bodies.index(
            "Pelvis"
        )
        self.target_pose_time[turning_envs] = (
                self.motion_times[turning_envs] + self.dt * time_left_to_turn[turning_envs]
        )
        self.target_pose_time[turned_envs] = (
                self.motion_times[turned_envs] + 1.0
        )  # .5 second
        # self.target_pose_obs_mask[:] = True
        self.target_pose_joints[:] = False
        self.target_pose_joints[turned_envs, head_body_index * 2] = True
        self.target_pose_joints[:, head_body_index * 2 + 1] = True
        new_mask = torch.zeros(
            self.num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.device,
        )
        new_mask[:, -1, :] = True  # heading and speed
        new_mask = new_mask.view(
            self.num_envs, 1, self.num_conditionable_bodies, 2
        ).expand(-1, self.config.masked_mimic_obs.num_future_steps, -1, -1)
        new_mask[:, -1, pelvis_body_index, 1] = True  # rotation
        new_mask[turned_envs, -1, pelvis_body_index, 0] = True  # translation
        new_mask = new_mask.reshape(self.num_envs, -1)
        self.masked_mimic_target_bodies_masks[:] = new_mask
        sparse_target_poses = self.build_sparse_target_heading_poses_masked_with_time(
            self.config.masked_mimic_obs.num_future_steps
        )
        self.masked_mimic_target_poses[:] = sparse_target_poses
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[turned_envs, 4:] = True

    ###############################################################
    # Helpers
    ###############################################################
    def build_sparse_target_heading_poses(self, raw_future_times):
        """
        This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
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
        cur_gt, cur_gr = current_state.body_pos, current_state.body_rot
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )
        # cur_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(self.num_envs, 1, 2)

        # override to set the target root parameters
        reshaped_target_pos = flat_target_pos.reshape(
            self.num_envs, num_future_steps, -1, 3
        )
        reshaped_target_rot = flat_target_rot.reshape(
            self.num_envs, num_future_steps, -1, 4
        )

        # turning_envs = self._heading_turn_steps > (self.progress_buf + 15)
        turning_envs = self._heading_turn_steps < 0
        turned_envs = ~turning_envs

        reshaped_target_pos[:, :, :, :2] = cur_gt[:, :, :2].unsqueeze(1).clone()
        for frame_idx in range(num_future_steps):
            reshaped_target_pos[:, frame_idx, :, :2] += (
                    self._tar_dir[:]
                    * self._tar_speed[:].unsqueeze(-1)
                    * (raw_future_times[:, frame_idx] - self.motion_times).unsqueeze(-1)
            ).unsqueeze(1)

        reshaped_target_pos[turning_envs, :, :, :2] = (
            cur_gt[turning_envs, :, :2].unsqueeze(1).clone()
        )

        reshaped_target_pos[:, :, 0, -1] = 0.88  # standing up
        reshaped_target_pos[:, :, 1:, -1] = 1.5  # head standing up

        angle = rotations.vec_to_heading(self._tar_facing_dir)
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        quat = rotations.heading_to_quat(angle, w_last=self.w_last)
        reshaped_target_rot[:, :, 0] = quat.unsqueeze(1)

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)
        flat_target_rot = reshaped_target_rot.reshape(flat_target_rot.shape)

        non_flat_target_vel = flat_target_vel.reshape(
            self.num_envs, num_future_steps, -1, 3
        )

        non_flat_target_vel[:, :, 0, :2] = (
                self._tar_dir[:] * self._tar_speed[:].unsqueeze(-1)
        ).view(self._tar_dir.shape[0], 1, 2)
        flat_target_vel = non_flat_target_vel.reshape(flat_target_vel.shape)
        # override to set the target root parameters

        expanded_body_pos = cur_gt.unsqueeze(1).expand(
            self.num_envs, num_future_steps, *cur_gt.shape[1:]
        )
        expanded_body_rot = cur_gr.unsqueeze(1).expand(
            self.num_envs, num_future_steps, *cur_gr.shape[1:]
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
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        padded_flat_target_body_pos = torch.nn.functional.pad(
            flat_target_body_pos, [0, 3], "constant", 0
        )
        sub_sampled_target_body_pos = padded_flat_target_body_pos.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        sub_sampled_target_rel_body_rot_obs = target_rel_body_rot_obs.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]
        sub_sampled_target_body_rot_obs = target_body_rot_obs.reshape(
            self.num_envs, num_future_steps, -1, 6
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
        ).reshape(self.num_envs, num_future_steps, 1, 6)

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
            self.num_envs, num_future_steps, 1, 6
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
        obs = torch.cat((obs, heading_and_velocity), dim=-2).view(self.num_envs, -1)

        return obs

    def build_sparse_target_heading_poses_masked_with_time(self, num_future_steps):
        time_offsets = (
                torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
                * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time.view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_heading_poses(all_future_times).view(
            self.num_envs,
            num_future_steps + 1,
            self.masked_mimic_conditionable_bodies_ids.shape[0] + 1,
            2,
            12,
        )

        near_mask = self.masked_mimic_target_bodies_masks.view(
            self.num_envs, num_future_steps, self.num_conditionable_bodies, 2, 1
        )
        far_mask = self.target_pose_joints.view(self.num_envs, 1, -1, 2, 1)
        mask = torch.cat([near_mask, far_mask], dim=1)

        masked_obs = obs * mask

        masked_obs_with_joints = torch.cat((masked_obs, mask), dim=-1).view(
            self.num_envs, num_future_steps + 1, -1
        )

        times = all_future_times.view(-1).view(
            self.num_envs, num_future_steps + 1, 1
        ) - self.motion_times.view(self.num_envs, 1, 1)
        ones_vec = torch.ones(
            self.num_envs, num_future_steps + 1, 1, device=self.device
        )
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(self.num_envs, -1)


@torch.jit.script
def compute_facing_observations(root_states, tar_face_dir, w_last: bool):
    root_rot = root_states[:, 3:7]
    tar_face_dir3d = torch.cat(
        [tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)
    local_tar_face_dir = rotations.quat_rotate(heading_rot, tar_face_dir3d, w_last)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]
    return local_tar_face_dir


@torch.jit.script
def compute_facing_reward(
        root_pos: Tensor,
        prev_root_pos: Tensor,
        root_rot: Tensor,
        tar_dir: Tensor,
        tar_speed: Tensor,
        tar_face_dir: Tensor,
        dt: float,
        w_last: bool,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    dir_reward, output_dict = compute_heading_reward(
        root_pos, prev_root_pos, tar_dir, tar_speed, dt
    )

    dir_reward_w = 0.7
    facing_reward_w = 0.3
    heading_rot = calc_heading_quat(root_rot, w_last)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir, w_last)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    output_dict["facing_dir"] = facing_dir
    output_dict["tar_face_dir"] = tar_face_dir
    output_dict["facing_err"] = facing_err
    output_dict["facing_reward"] = facing_reward

    return reward, output_dict
