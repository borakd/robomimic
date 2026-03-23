import os
import json
import h5py
import numpy as np
import robosuite
import robosuite.utils.camera_utils as camera_utils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

# Base directory where your datasets are stored
base_dir = "/media/bora/Extreme Pro"

# The exact camera mappings
task_cameras = {
    "can": ["agentview", "robot0_eye_in_hand"],
    "lift": ["agentview", "robot0_eye_in_hand"],
    "square": ["agentview", "robot0_eye_in_hand"],
    "tool_hang": ["robot0_eye_in_hand", "sideview"],
    "transport": ["robot0_eye_in_hand", "robot1_eye_in_hand", "shouldercamera0", "shouldercamera1"]
}

# The resolution you rendered at
HEIGHT = 512
WIDTH = 512

for task, camera_names in task_cameras.items():
    dataset_path = os.path.join(base_dir, task, "ph", "image_depth_512.hdf5")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        continue
        
    print(f"--- Processing camera info for Task: {task.upper()} ---")
    
    # 1. Read env_meta to reconstruct the robosuite environment
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # Ensure our desired cameras and resolution are set
    env_meta["env_kwargs"]["camera_names"] = camera_names
    env_meta["env_kwargs"]["camera_heights"] = HEIGHT
    env_meta["env_kwargs"]["camera_widths"] = WIDTH
    
    # Create the environment
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, env_name=env_meta["env_name"], render=False, render_offscreen=True)
    
    with h5py.File(dataset_path, "r+") as f:
        demos = list(f["data"].keys())
        
        for ep_idx, ep in enumerate(demos):
            ep_group = f["data"][ep]
            states = ep_group["states"][:]  # The raw MuJoCo states
            num_steps = len(states)
            
            # Prepare storage for this episode
            if "camera_info" not in ep_group["obs"]:
                ep_group["obs"].create_group("camera_info")
                
            for cam in camera_names:
                if cam not in ep_group["obs"]["camera_info"]:
                    ep_group["obs"]["camera_info"].create_group(cam)
                    
                # Intrinsics are static, so shape is (3, 3)
                if "intrinsics" not in ep_group["obs"]["camera_info"][cam]:
                    ep_group["obs"]["camera_info"][cam].create_dataset("intrinsics", shape=(3, 3), dtype=np.float32)
                    
                # Extrinsics are DYNAMIC, so shape is (num_steps, 4, 4)
                if "extrinsics" not in ep_group["obs"]["camera_info"][cam]:
                    ep_group["obs"]["camera_info"][cam].create_dataset("extrinsics", shape=(num_steps, 4, 4), dtype=np.float32)
            
            # Initialize arrays to hold the timestep data
            extrinsics_buffer = {cam: np.zeros((num_steps, 4, 4)) for cam in camera_names}
            intrinsics_buffer = {cam: np.zeros((3, 3)) for cam in camera_names}
            
            # 2. Replay the states and extract dynamic extrinsics
            for t in range(num_steps):
                # Set MuJoCo to the exact state at this timestep
                env.env.sim.set_state_from_flattened(states[t])
                env.env.sim.forward() # Update kinematics (crucial for moving the eye-in-hand camera)
                
                for cam in camera_names:
                    # Get 3x3 rotation matrix and 3D position vector
                    cam_pos = env.env.sim.data.get_camera_xpos(cam)
                    cam_rot = env.env.sim.data.get_camera_xmat(cam)
                    
                    # Construct 4x4 homogeneous extrinsic matrix
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = cam_rot
                    extrinsic[:3, 3] = cam_pos
                    extrinsics_buffer[cam][t] = extrinsic
                    
                    # Get static intrinsic matrix (only need to calculate once per episode)
                    if t == 0:
                        intrinsics_buffer[cam] = camera_utils.get_camera_intrinsic_matrix(
                            sim=env.env.sim, camera_name=cam, camera_height=HEIGHT, camera_width=WIDTH
                        )
            
            # 3. Save into HDF5
            for cam in camera_names:
                ep_group["obs"]["camera_info"][cam]["extrinsics"][:] = extrinsics_buffer[cam]
                ep_group["obs"]["camera_info"][cam]["intrinsics"][:] = intrinsics_buffer[cam]
                
            if (ep_idx + 1) % 10 == 0:
                print(f"Processed {ep_idx + 1}/{len(demos)} demonstrations...")
                
    print(f"Finished {task}!\n")