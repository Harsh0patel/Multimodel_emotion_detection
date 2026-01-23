import os
import cv2
import pandas as pd
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm

def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_single_video(args):
    video_path, face_cascade_path = args

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    cap = cv2.VideoCapture(video_path)

    video_name = os.path.basename(video_path)
    frame_data = []
    selected_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        PROCESS_EVERY_N = 5  # or 10

        if frame_idx % PROCESS_EVERY_N != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        metrics = None
        quality_label = 0

        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda x: x[2] * x[3])

            metrics = {
                "face_area": (fw * fh) / (w * h),
                "blur_score": calculate_blur(gray[y:y+fh, x:x+fw]),
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w": int(fw),
                "bbox_h": int(fh)
            }
        
            frame_data.append({
                "video_name": video_name,
                "frame_idx": frame_idx,
                "quality_label": quality_label,
                "face_area": (fw * fh) / (w * h),
                "blur_score": calculate_blur(gray[y:y+fh, x: x+fw]),
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w": int(fw),
                "bbox_h": int(fh),
                "metrics": metrics
            })

        frame_idx += 1

    # ---- TOP-X% GATING LOGIC ----

    # keep only frames with valid metrics
    valid_frames = [
    f for f in frame_data
    if f["metrics"] is not None
    ]

    if len(valid_frames) > 0:
        # compute quality score
        for f in valid_frames:
            f["quality_score"] = (
               0.6 * f["metrics"]["blur_score"] + 
               0.4 * f["metrics"]["face_area"]
            )

        # sort by quality
        valid_frames.sort(
            key=lambda x: x["quality_score"],
            reverse=True
        )

        # select top X%
        TOP_PERCENT = 0.20
        MIN_FRAMES = 3

        k = max(MIN_FRAMES, int(len(valid_frames) * TOP_PERCENT))
        selected = valid_frames[:k]

        # assign labels
        selected_ids = {f["frame_idx"] for f in selected}

        for f in frame_data:
            if f["frame_idx"] in selected_ids:
                f["quality_label"] = 1
                selected_frames.append(f)

    cap.release()
    return selected_frames


def process_meld_parallel(meld_video_dir, output_csv, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    video_files = [f for f in os.listdir(meld_video_dir) if f.endswith(".mp4")]
    video_paths = [os.path.join(meld_video_dir, f) for f in video_files]

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    args_list = [(vp, face_cascade_path) for vp in video_paths]

    all_results = []

    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_video, args_list),
            total=len(args_list),
            desc="Processing videos"
        ):
            all_results.extend(result)

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)

    print(f"✓ Videos: {len(video_files)}")
    print(f"✓ Frames: {len(df)}")
    print(f"✓ Good frames: {df['quality_label'].sum()}")

    return df 

if __name__ == "__main__":
    set_start_method("spawn", force=True)

    df = process_meld_parallel(
        meld_video_dir="C:/Users/hp333/Desktop/Multimodel_emotion_detection/data/MELD.Raw/train/train_splits",
        output_csv="meld_face_quality.csv",
        num_workers=12
    )

