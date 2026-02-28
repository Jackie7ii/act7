"""
Convert LeRobot format (Parquet + Video) to ACT format (HDF5).

LeRobot format:
  data/chunk-000/file-000.parquet  → state, action, episode_index, frame_index
  videos/observation.images.{cam}/chunk-000/file-000.mp4  → all episodes packed in one video

ACT HDF5 format (per episode):
  episode_{idx}.hdf5
    /observations/qpos       (episode_len, 14)
    /observations/qvel       (episode_len, 14)   ← zeros (not in LeRobot)
    /observations/images/{cam_name}  (episode_len, H, W, 3)
    /action                  (episode_len, 14)
    attrs['sim'] = False

Usage:
  python convert_lerobot_to_hdf5.py \
      --lerobot_dir aloha_static_cups_open \
      --output_dir /path/to/output/hdf5 \
      --camera_names cam_high \
      --num_episodes 50
"""

import os
import argparse
import numpy as np
import pandas as pd
import h5py
import av
from tqdm import tqdm


# LeRobot camera key prefix
LEROBOT_CAM_PREFIX = "observation.images."


def extract_frames_from_video(video_path, start_frame, num_frames):
    """
    Extract a range of frames from a video file using PyAV.
    Returns numpy array of shape (num_frames, H, W, 3) in uint8.
    """
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]

        # Seek to approximate position (may not be exact for all codecs)
        # We'll decode from the start for small num_frames, or use seek for large offset
        frame_idx = 0
        target_end = start_frame + num_frames

        for frame in container.decode(stream):
            if frame_idx >= target_end:
                break
            if frame_idx >= start_frame:
                img = frame.to_ndarray(format='rgb24')  # (H, W, 3)
                frames.append(img)
            frame_idx += 1

    if len(frames) != num_frames:
        raise ValueError(
            f"Expected {num_frames} frames but got {len(frames)} "
            f"(start={start_frame}, video={video_path})"
        )
    return np.stack(frames, axis=0)  # (num_frames, H, W, 3)


def convert(lerobot_dir, output_dir, camera_names, num_episodes):
    os.makedirs(output_dir, exist_ok=True)

    # Load parquet (all episodes in one file)
    parquet_path = os.path.join(lerobot_dir, "data", "chunk-000", "file-000.parquet")
    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # Map camera short names to LeRobot video paths
    video_paths = {}
    for cam in camera_names:
        lerobot_key = LEROBOT_CAM_PREFIX + cam  # e.g. "observation.images.cam_high"
        vpath = os.path.join(lerobot_dir, "videos", lerobot_key, "chunk-000", "file-000.mp4")
        if not os.path.exists(vpath):
            raise FileNotFoundError(f"Video not found: {vpath}\n"
                                    f"Available cameras: cam_high, cam_left_wrist, cam_low, cam_right_wrist")
        video_paths[cam] = vpath
        print(f"  Camera '{cam}' → {vpath}")

    # Convert each episode
    for ep_idx in tqdm(range(num_episodes), desc="Converting episodes"):
        ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        episode_len = len(ep_df)

        if episode_len == 0:
            print(f"  WARNING: Episode {ep_idx} has no data, skipping.")
            continue

        # Extract state and action
        qpos = np.stack(ep_df["observation.state"].values).astype(np.float32)  # (T, 14)
        action = np.stack(ep_df["action"].values).astype(np.float32)            # (T, 14)
        qvel = np.zeros_like(qpos)                                              # (T, 14) zeros

        # Global frame offset in the monolithic video
        global_start_frame = ep_idx * episode_len

        # Extract frames for each camera
        cam_frames = {}
        for cam in camera_names:
            frames = extract_frames_from_video(
                video_paths[cam], global_start_frame, episode_len
            )  # (T, H, W, 3)
            cam_frames[cam] = frames

        # Write HDF5
        hdf5_path = os.path.join(output_dir, f"episode_{ep_idx}.hdf5")
        with h5py.File(hdf5_path, "w") as root:
            root.attrs["sim"] = False

            obs_grp = root.create_group("observations")
            obs_grp.create_dataset("qpos", data=qpos)
            obs_grp.create_dataset("qvel", data=qvel)

            img_grp = obs_grp.create_group("images")
            for cam in camera_names:
                img_grp.create_dataset(
                    cam, data=cam_frames[cam],
                    dtype="uint8",
                    chunks=(1, *cam_frames[cam].shape[1:]),  # chunk per frame for fast random access
                    compression="gzip", compression_opts=2
                )

            root.create_dataset("action", data=action)

    print(f"\nDone! {num_episodes} episodes saved to: {output_dir}")
    print(f"Camera names to use in training: {camera_names}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot_dir", type=str,
                        default="aloha_static_cups_open",
                        help="Path to the cloned LeRobot dataset directory")
    parser.add_argument("--output_dir", type=str,
                        default="data/aloha_cups_open_hdf5",
                        help="Output directory for HDF5 files")
    parser.add_argument("--camera_names", type=str, nargs="+",
                        default=["cam_high"],
                        help="Camera(s) to include. Choices: cam_high, cam_left_wrist, cam_low, cam_right_wrist")
    parser.add_argument("--num_episodes", type=int, default=50)
    args = parser.parse_args()

    convert(
        lerobot_dir=args.lerobot_dir,
        output_dir=args.output_dir,
        camera_names=args.camera_names,
        num_episodes=args.num_episodes,
    )


if __name__ == "__main__":
    main()
