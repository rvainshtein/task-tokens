from typing import Optional

import numpy as np
import torch
from isaac_utils import torch_utils
from isaacgym import gymapi, gymtorch  # type: ignore[misc]

from phys_anim.envs.masked_mimic_inversion.base_task.isaacgym import (
    MaskedMimicTaskHumanoid,
)
from phys_anim.envs.masked_mimic_inversion.path_follower.common import (
    BaseMaskedMimicPathFollowing,
)
from phys_anim.utils.motion_lib import MotionLib


class MaskedMimicPathFollowingHumanoid(BaseMaskedMimicPathFollowing, MaskedMimicTaskHumanoid):  # type: ignore[misc]
    def __init__(
        self, config, device: torch.device, motion_lib: Optional[MotionLib] = None
    ):
        super().__init__(config=config, device=device, motion_lib=motion_lib)

        if not self.headless:
            self._build_marker_state_tensors()

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        super().create_envs(num_envs, spacing, num_per_row)

    def _load_marker_asset(self):
        asset_root = "phys_anim/data/assets/urdf/"
        asset_file = "traj_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)

        if not self.headless:
            self._build_marker(env_id, env_ptr)

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        for i in range(self.config.path_follower_params.num_traj_samples):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                marker_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.8, 0.0, 0.0),
            )
            self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self.root_states.shape[0] // self.num_envs
        self._marker_states = self.root_states.view(
            self.num_envs, num_actors, self.root_states.shape[-1]
        )[..., 1 : (1 + self.config.path_follower_params.num_traj_samples), :]
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
        traj_samples = self.fetch_path_samples(time_offset=0)[0].clone()
        self._marker_pos[:] = traj_samples
        if not self.config.path_follower_params.path_generator.height_conditioned:
            self._marker_pos[..., 2] = 0.8  # CT hack

        ground_below_marker = self.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])

        self._marker_pos[..., 2] += ground_below_marker

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

    def draw_task(self):
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # bodies_positions = self.get_body_positions()

        self._update_marker()

        for i, env_ptr in enumerate(self.envs):
            verts = self.path_generator.get_traj_verts(i).clone()
            if not self.config.path_follower_params.path_generator.height_conditioned:
                verts[..., 2] = self.humanoid_root_states[i, 2]  # ZL Hack
            else:
                verts[..., 2] += self.get_ground_heights(
                    self.humanoid_root_states[i, :2].view(1, 2)
                ).view(-1)
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)
