import os
import subprocess

# Base directory where your datasets are stored
base_dir = "/media/bora/Extreme Pro/Eren/tasks"

# The exact camera mappings you provided
task_cameras = {
    "can": ["agentview", "robot0_eye_in_hand"],
    "lift": ["agentview", "robot0_eye_in_hand"],
    "square": ["agentview", "robot0_eye_in_hand"],
    "tool_hang": ["robot0_eye_in_hand", "sideview"],
    "transport": ["robot0_eye_in_hand", "robot1_eye_in_hand", "shouldercamera0", "shouldercamera1"]
}

for task, camera_names in task_cameras.items():
    dataset_path = os.path.join(base_dir, task, "ph", "demo_v15.hdf5")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        continue
        
    print(f"--- Task: {task.upper()} ---")
    print(f"Using cameras: {camera_names}")
    
    # Build the command for dataset_states_to_obs.py matching your local arguments
    cmd = [
        "python", "robomimic/scripts/dataset_states_to_obs.py",
        "--dataset", dataset_path,
        "--output_name", "image_depth_512.hdf5",
        "--done_mode", "2",
        "--camera_height", "512",
        "--camera_width", "512",
        "--depth",  # Using the correct flag for your version
        "--camera_names"
    ] + camera_names
    
    # Execute the extraction script
    print("Executing extraction...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully finished processing {task}.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing {task}: {e}\n")