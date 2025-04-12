from typing import Optional

import torch

# from phys_anim.envs.masked_mimic_inversion.base_task.isaacgym import MaskedMimicTaskHumanoid
from phys_anim.envs.masked_mimic_inversion.direction_facing.common import (
    MaskedMimicBaseDirectionFacing,
)
from phys_anim.envs.masked_mimic_inversion.steering.isaacgym import (
    MaskedMimicDirectionHumanoid,
)
from isaacgym import gymapi, gymtorch  # type: ignore[misc]
from isaac_utils import rotations

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2


class MaskedMimicDirectionFacingHumanoid(
    MaskedMimicBaseDirectionFacing, MaskedMimicDirectionHumanoid
):
    def __init__(
        self, config, device: torch.device, motion_lib: Optional[torch.Tensor] = None
    ):
        super().__init__(config=config, device=device)

        if not self.headless:
            self._build_marker_state_tensors()

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = []
            self._face_marker_handles = []
            self._load_marker_asset()

        super().create_envs(num_envs, spacing, num_per_row)

    def _load_marker_asset(self):
        asset_root = "phys_anim/data/assets/urdf/"
        asset_file = "heading_marker.urdf"

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
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        default_pose.p.z = 0.0

        marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "marker",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0)
        )
        self._marker_handles.append(marker_handle)

        face_marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "face_marker",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr,
            face_marker_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.0, 0.0, 0.8),
        )
        self._face_marker_handles.append(face_marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self.root_states.shape[0] // self.num_envs

        self._marker_states = self.root_states.view(
            self.num_envs, num_actors, self.root_states.shape[-1]
        )[..., TAR_ACTOR_ID, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self.humanoid_actor_ids + TAR_ACTOR_ID

        self._face_marker_states = self.root_states.view(
            self.num_envs, num_actors, self.root_states.shape[-1]
        )[..., TAR_FACING_ACTOR_ID, :]
        self._face_marker_pos = self._face_marker_states[..., :3]
        self._face_marker_rot = self._face_marker_states[..., 3:7]
        self._face_marker_actor_ids = self.humanoid_actor_ids + TAR_FACING_ACTOR_ID

        return

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        humanoid_root_pos = self.get_humanoid_root_states()[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_dir
        self._marker_pos[..., 2] = humanoid_root_pos[..., 2]

        heading_theta = (
            self._tar_dir_theta
        )  # torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        heading_q = rotations.quat_from_angle_axis(
            heading_theta, heading_axis, w_last=True
        )
        self._marker_rot[:] = heading_q

        self._face_marker_pos[..., 0:2] = (
            humanoid_root_pos[..., 0:2] + self._tar_facing_dir
        )
        self._face_marker_pos[..., 2] = humanoid_root_pos[..., 2]

        face_theta = torch.atan2(
            self._tar_facing_dir[..., 1], self._tar_facing_dir[..., 0]
        )
        face_axis = torch.zeros_like(self._marker_pos)
        face_axis[..., -1] = 1.0
        face_q = rotations.quat_from_angle_axis(face_theta, heading_axis, w_last=True)
        self._face_marker_rot[:] = face_q

        marker_ids = torch.cat(
            [self._marker_actor_ids, self._face_marker_actor_ids], dim=0
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(marker_ids),
            len(marker_ids),
        )

    def draw_task(self):
        self._update_marker()

        # # Get the humanoid root position
        # humanoid_root_pos = self.get_humanoid_root_states()[..., 0:3].cpu().numpy()
        #
        # # Get facing direction (target direction)
        # facing_dir = self._tar_facing_dir.cpu().numpy()
        #
        # # Get actual moving direction (based on velocity)
        # velocity = self.get_humanoid_velocity()  # This function needs to return the velocity in the x, y plane
        # moving_dir = velocity[..., :2]  # Assuming velocity is 3D, use the X and Y components for 2D direction
        # moving_dir /= torch.norm(moving_dir, dim=-1, keepdim=True)  # Normalize to get unit vector
        #
        # # Get speed (magnitude of the velocity vector)

    #     speed = torch.norm(velocity, dim=-1).cpu().numpy()
    #
    #     # for i in range(self.num_envs):
    #     start = humanoid_root_pos
    #
    #     # Facing direction vector (purple)
    #     end_facing = start + 0.5 * facing_dir  # Adjust length (0.5 here is arbitrary)
    #     self.gym.add_lines(
    #         self.envs[0],
    #         1,
    #         [gymapi.Vec3(*start), gymapi.Vec3(*end_facing)],
    #         [gymapi.Vec3(0.5, 0.0, 0.5), gymapi.Vec3(0.5, 0.0, 0.5)],  # Purple
    #     )
    #
    #     # Moving direction vector (green)
    #     end_moving = start + 0.5 * moving_dir  # Adjust length (0.5 here is arbitrary)
    #     self.gym.add_lines(
    #         self.envs[0],
    #         1,
    #         [gymapi.Vec3(*start), gymapi.Vec3(*end_moving)],
    #         [gymapi.Vec3(0.0, 1.0, 0.0), gymapi.Vec3(0.0, 1.0, 0.0)],  # Green
    #     )
    #
    #     # Speed annotation near the moving direction vector
    #     text_position = gymapi.Vec3(*end_moving)
    #     self.gym.draw_text(
    #         self.viewer,
    #         f"{speed:.2f}",
    #         text_position,
    #     )
    #
    # def get_humanoid_velocity(self):
    #     # Assuming you have access to the velocity in the state tensor (e.g., from simulation or robot model)
    #     velocity = self.root_states[..., 7:10]  # Assuming the velocity is in indices 7:10
    #     return velocity
