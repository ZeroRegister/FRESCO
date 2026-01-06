import os
import sys
import glob
import argparse
import yaml
import torch
import torch.distributed as dist
from run_fresco import get_models, run_keyframe_translation, run_full_video_translation

def setup_ddp():
    """Initialize DDP environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        # Fallback for single GPU/testing
        rank = 0
        world_size = 1
        local_rank = 0
        print("DDP environment not detected. Running in single-process mode.")
    
    return rank, world_size, local_rank

def get_video_prompt_pairs(video_folder, prompt_folder):
    """
    Find matching video and prompt files.
    Assumes video.mp4 corresponds to video.txt.
    """
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(glob.glob(os.path.join(video_folder, ext)))
    
    all_videos.sort()
    pairs = []
    for video_path in all_videos:
        basename = os.path.splitext(os.path.basename(video_path))[0]
        prompt_path = os.path.join(prompt_folder, basename + ".txt")
        
        if os.path.exists(prompt_path):
            pairs.append((video_path, prompt_path))
        else:
            print(f"Warning: No matching prompt found for {video_path}, skipping.")
            
    return pairs

def main():
    parser = argparse.ArgumentParser(description="DDP Inference for Sim2Real FRESCO")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing input videos")
    parser.add_argument("--prompt_folder", type=str, required=True, help="Folder containing matching prompt txt files")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base config yaml")
    parser.add_argument("--output_folder", type=str, required=True, help="Root folder for outputs")
    parser.add_argument("--control_type", type=str, default="hed", choices=["hed", "canny", "depth"], help="ControlNet type")
    
    args = parser.parse_args()
    
    # 1. Setup DDP
    rank, world_size, local_rank = setup_ddp()
    if rank == 0:
        print(f"Starting generic FRESCO Sim2Real Inference with {world_size} GPUs.")

    # 2. Prepare Data (Only Rank 0 needs to scan, but for simplicity everyone scans locally)
    # We ensure deterministic sort so all ranks see the same list
    work_items = get_video_prompt_pairs(args.video_folder, args.prompt_folder)
    
    # Partition work
    my_items = work_items[rank::world_size]
    
    if len(my_items) == 0:
        print(f"Rank {rank}: No items assigned. Exiting.")
        return

    print(f"Rank {rank} has assigned {len(my_items)} videos.")

    # 3. Load Configuration
    with open(args.base_config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override control type if specified
    config['controlnet_type'] = args.control_type
    
    # 4. Load Models (Once per process)
    # IMPORTANT: get_models puts models on 'cuda'. 
    # Since we called set_device(local_rank), 'cuda' refers to the correct device.
    print(f"Rank {rank}: Loading models...")
    models = get_models(config)
    print(f"Rank {rank}: Models loaded.")

    # 5. Inference Loop
    for video_path, prompt_path in my_items:
        # Read prompt
        with open(prompt_path, 'r') as f:
            prompt_text = f.read().strip()
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_save_path = os.path.join(args.output_folder, video_name) + os.sep
        
        print(f"Rank {rank}: Processing {video_name}...")
        
        # Prepare specific config for this video
        current_config = config.copy()
        current_config['file_path'] = video_path
        current_config['save_path'] = video_save_path
        current_config['prompt'] = prompt_text
        
        try:
            # Run Keyframe Translation
            # Pass pre-loaded models to avoid reloading
            keys = run_keyframe_translation(current_config, models=models)
            
            # Run Full Video Translation (EBSynth blending)
            # This part is largely CPU based but orchestrated here
            run_full_video_translation(current_config, keys)
            
        except Exception as e:
            print(f"Rank {rank}: Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Rank {rank}: Finished processing all assigned videos.")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
