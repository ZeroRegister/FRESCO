#!/usr/bin/env python
"""
Simple single-GPU inference script for FRESCO video translation.
Usage:
    python inference_simple.py --video ./data/car-turn.mp4 --prompt "a red car turns in the winter" --output ./output/car-turn/
"""

import os
import sys
import argparse
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_fresco import get_models, run_keyframe_translation, run_full_video_translation


def main():
    parser = argparse.ArgumentParser(description="Simple FRESCO Video Translation")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for translation")
    parser.add_argument("--output", type=str, required=True, help="Output directory (will be created)")
    parser.add_argument("--config", type=str, default="./config/config_carturn.yaml", 
                        help="Base config file (default: config_carturn.yaml)")
    parser.add_argument("--control_type", type=str, default="hed", 
                        choices=["hed", "canny", "depth"], help="ControlNet type")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no_ebsynth", action="store_true", 
                        help="Skip EBSynth blending (only generate keyframes)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Load base config
    print("=" * 60)
    print("Loading configuration...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    config['file_path'] = os.path.abspath(args.video)
    config['prompt'] = args.prompt
    config['save_path'] = os.path.abspath(args.output)
    if not config['save_path'].endswith(os.sep):
        config['save_path'] += os.sep
    config['controlnet_type'] = args.control_type
    config['seed'] = args.seed
    config['run_ebsynth'] = not args.no_ebsynth
    
    # Print config
    print("=" * 60)
    print("Configuration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(config['save_path'], exist_ok=True)
    
    # Step 1: Run keyframe translation (diffusion model)
    print("\n[Step 1/2] Running keyframe translation...")
    keys = run_keyframe_translation(config)
    print(f"Generated {len(keys)} keyframes: {keys}")
    
    # Step 2: Run full video translation (EBSynth blending)
    print("\n[Step 2/2] Running full video translation (EBSynth)...")
    run_full_video_translation(config, keys)
    
    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Output directory: {config['save_path']}")
    print(f"  Keyframes: {config['save_path']}keys/")
    if config['run_ebsynth']:
        print(f"  Final video: {config['save_path']}blend.mp4")
    else:
        print("  (EBSynth skipped - run video_blend.py manually)")
    print("=" * 60)


if __name__ == "__main__":
    main()
