import time

import cv2
import math
import numpy as np
from pyvirtualdisplay.smartdisplay import SmartDisplay

# Start a virtual display to simulate a monitor on a headless server.
# For headless, visible should be False.
display_width = 1280
display_height = 720
display = SmartDisplay(visible=False, size=(display_width, display_height))
display.start()

from isaacgym import gymapi

# Acquire the gym API.
gym = gymapi.acquire_gym()

# Set up simulation parameters.
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z  # Use Z-up coordinate system.
# For stability on headless setups, we disable GPU pipeline (you can try True if your setup is reliable)
sim_params.use_gpu_pipeline = False

# Create the simulation (using compute device 0 and graphics device 0).
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Add a ground plane.
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # For Z-up, normal is (0, 0, 1).
gym.add_ground(sim, plane_params)

# Create one environment.
env_lower = gymapi.Vec3(-5, 0, -5)
env_upper = gymapi.Vec3(5, 5, 5)
env = gym.create_env(sim, env_lower, env_upper, 1)

# Load the Humanoid asset from an MJCF file.
asset_root = "../data/assets"  # Update this path to your asset folder.
asset_file = "mjcf/smpl_humanoid.xml"  # Use the MJCF file for the humanoid.
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False  # Humanoid should be dynamic.
humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Create a humanoid actor with an initial pose.
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 1, 0)  # Position the humanoid 1 meter above the ground.
pose.r = gymapi.Quat(0, 0, 0, 1)
humanoid_actor = gym.create_actor(env, humanoid_asset, pose, "humanoid", 0, 1)

# Create a camera sensor.
cam_props = gymapi.CameraProperties()
cam_props.width = display_width
cam_props.height = display_height
# Create the sensor using the environment and camera properties.
camera_handle = gym.create_camera_sensor(env, cam_props)
# Set the camera's transform.
cam_pose = gymapi.Transform()
cam_pose.p = gymapi.Vec3(3, 3, 3)  # Position the camera.
cam_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 4)
gym.set_camera_transform(camera_handle, env, cam_pose)  # (camera_handle, env, cam_pose)

# ---- Create a viewer so that something is actually rendered to the display ----
# We'll use the same camera properties for the viewer.
viewer = gym.create_viewer(sim, cam_props)
if viewer is None:
    raise Exception("Failed to create viewer. Make sure your virtual display is working.")

images = []
num_steps = 300  # Adjust for a longer video.
for i in range(num_steps):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    # Update the viewer so that the window gets drawn on the virtual display.
    gym.draw_viewer(viewer, sim, True)

    # Optionally, you can add a small sleep to ensure the frame is drawn.
    time.sleep(0.01)

    # Grab the frame from the virtual display.
    img = display.grab()
    color_image = np.array(img)

    if color_image.size == 0:
        print(f"Step {i}: No image captured!")
        continue

    # Get image shape; if empty, skip frame.
    try:
        H, W, C = color_image.shape
    except Exception as e:
        print(f"Step {i}: Error unpacking image shape: {e}")
        continue
    annoying_window_crop = int(W * 0.25)
    # Convert from RGB to BGR for OpenCV.
    color_image = color_image[:, annoying_window_crop:, :]
    image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    images.append(image_bgr)

# Set up video writer.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 30  # Desired frame rate.
video_width, video_height = images[0].shape[1], images[0].shape[0]
video_writer = cv2.VideoWriter("humanoid_video.mp4", fourcc, fps, (video_width, video_height))
for image_bgr in images:
    video_writer.write(image_bgr)

video_writer.release()
gym.destroy_sim(sim)
display.stop()
