import os
import subprocess
import glob

def frame_to_video(video_path, frame_dir, fps, verbose=False):
    """
    Compiles frames in a directory into a video using ffmpeg.
    Assumes frames are named %04d.png.
    """
    if verbose:
        print(f"Compiling video to {video_path} from {frame_dir} at {fps} fps")
    
    # Check for empty directory
    if not os.path.exists(frame_dir) or not os.listdir(frame_dir):
        print(f"Warning: Frame directory {frame_dir} is empty or does not exist.")
        return

    # Pattern assumption: FRESCO uses %04d.png
    pattern = os.path.join(frame_dir, "%04d.png")
    
    # ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        video_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=None if verbose else subprocess.DEVNULL, stderr=None if verbose else subprocess.DEVNULL)
        if verbose:
            print(f"Successfully saved video to {video_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        # Try glob approach if pattern fails (e.g. missing frames in sequence)
        # But FRESCO EBSynth usually produces full sequence
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
