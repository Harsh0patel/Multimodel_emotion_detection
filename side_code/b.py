import os
import cv2
from tqdm import tqdm

def extract_from_all_videos(video_dir, output_base_dir, target_fps=6):
    """
    Extract frames from all videos in a directory
    
    Args:
        video_dir: directory containing videos
        output_base_dir: base directory for all extracted frames
        target_fps: frames per second to extract
    """
    
    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files = video_files[:1500]
    
    print(f"Found {len(video_files)} videos")
    
    for video_name in tqdm(video_files, desc="Extracting frames"):
        video_path = os.path.join(video_dir, video_name)
        
        # Create output dir for this video
        video_basename = os.path.splitext(video_name)[0]
        output_dir = os.path.join(output_base_dir, video_basename)
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / target_fps))
        
        frame_idx = 0
        saved_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
            
            frame_idx += 1
        
        cap.release()
    
    print(f"\n✓ Extraction complete!")

# Extract from all MELD videos at 6 fps
extract_from_all_videos(
    video_dir='C:/Users/hp333/Desktop/Multimodel_emotion_detection/data/MELD.Raw/train/train_splits',
    output_base_dir='meld_all_frames',
    target_fps=2
)