import os
import multiprocessing as mp
from pathlib import Path
import argparse
import imageio
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def pad_video(video_path, target_length, output_path):
    video = imageio.get_reader(video_path)
    frames = [frame for frame in video]
    current_length = len(frames)

    if current_length < target_length:
        first_frame = frames[0]
        padding_frames = [first_frame] * (target_length - current_length)
        frames = padding_frames + frames
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=video.get_meta_data()['fps'])

    if "merged" in str(video_path) and "pose" not in str(video_path):
        low_dim_file = str(video_path).replace("merged", "low_dim").replace(".mp4", ".npz")
        low_dim_output_path = output_path.parent / Path(low_dim_file).name
        low_dim_data = dict(np.load(low_dim_file, allow_pickle=True))
        keys = ["actions", "states", "ee_states", "camera_poses"]
        for key in keys:
            data = low_dim_data[key]
            if len(data) < target_length:
                padding_data = np.repeat(data[:1], target_length - len(data), axis=0)
                data = np.concatenate([padding_data, data], axis=0)
            low_dim_data[key] = data
        np.savez(low_dim_output_path, **low_dim_data)

def find_video_files(input_dir):
    input_dir = Path(input_dir)
    video_files = list(input_dir.rglob('*.mp4'))
    video_files = [video_file for video_file in video_files if "seg0" in video_file.name]
    return video_files


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Pad videos to a target length.")
    args.add_argument("--input_dir", type=str, required=True, help="Directory containing input videos.")
    args.add_argument("--target_length", type=int, required=True, help="Target number of frames for each video.")
    args.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers to use.")
    args = args.parse_args()

    input_dir = Path(args.input_dir)
    if "len" in str(input_dir):
        length_ori = int(str(input_dir).split("len")[1].split("_")[0])
        output_dir = str(input_dir).replace(f"len{length_ori}", f"len{args.target_length}")
    output_dir = Path(output_dir + "_static_prefix")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = find_video_files(args.input_dir)
    tasks = [(video_file, args.target_length, output_dir / video_file.relative_to(args.input_dir)) for video_file in video_files]

    print(f"Found {len(video_files)} video files to process.")
    
    # Process videos in parallel with progress bar using batched submission
    batch_size = min(args.num_workers * 2, 100)  # Limit batch size to avoid memory issues
    
    with tqdm(total=len(tasks), desc="Processing videos") as pbar:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Process tasks in batches to avoid overwhelming the executor
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                future_to_task = {}
                
                # Submit current batch
                for task in batch:
                    future = executor.submit(pad_video, *task)
                    future_to_task[future] = task
                
                # Process completed tasks from current batch
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()  # This will raise an exception if the task failed
                        pbar.set_postfix({"current": os.path.basename(task[0])})
                    except Exception as exc:
                        print(f"\nVideo {task[0]} generated an exception: {exc}")
                    finally:
                        pbar.update(1)
    
    print(f"Processing complete! Output saved to {output_dir}")

    