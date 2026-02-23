import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
import torchvision.models as models
from ultralytics.models.yolo import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Process 32 faces at once
model_path = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/ravdess_model/model/yolov8n-face.pt"
base_path = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/ravdess_model/data"
splits = ["Train", "Test", "Dev"]

print(f"Using device: {DEVICE}")

# Load ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()
for params in model.parameters():
    params.requires_grad = False
model = model.to(DEVICE)
model.eval()

# Load YOLO
yolo_model = YOLO(model_path)

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def make_embeddings_batch(frame_list):
    """Batch process PIL images for faster embedding generation"""
    if len(frame_list) == 0:
        return np.zeros(2048, dtype=np.float32)
    
    all_features = []
    
    # Process in batches of BATCH_SIZE
    for i in range(0, len(frame_list), BATCH_SIZE):
        batch_frames = frame_list[i:i+BATCH_SIZE]
        
        # Stack batch and send to device
        batch_tensor = torch.stack([
            transform(face) for face in batch_frames
        ]).to(DEVICE)
        
        # Forward pass
        with torch.no_grad():
            features = model(batch_tensor)
            features = features.cpu().numpy().astype(np.float32)
            all_features.extend(features)
    
    # Return mean embedding
    return np.mean(all_features, axis=0)

def process_split(split_name):
    """Process all videos in a split"""
    input_dir = os.path.join(base_path, split_name)
    output_dir = os.path.join(base_path, split_name, "embeddings", "Video")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {split_name} data...")
    
    video_count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if not file.endswith('.mp4'):
                continue
            
            file_path = os.path.join(root, file)
            cap = cv2.VideoCapture(file_path)
            temp = []
            
            try:
                while True:
                    res, frame = cap.read()
                    if not res:
                        break

                    # Detect faces
                    result = yolo_model(frame, conf=0.7, verbose=False)
                    
                    # Extract faces
                    for box in result[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cropped_face = frame[y1:y2, x1:x2]
                        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(cropped_face_rgb)
                        temp.append(pil_image)
                
                cap.release()

                # Generate embeddings with batch processing
                output_filename = file.replace('.mp4', '.npy')
                npy_array = make_embeddings_batch(temp)
                np.save(os.path.join(output_dir, output_filename), npy_array)
                
                video_count += 1
            
            except Exception as e:
                print(f"Error processing {file}: {e}")
                cap.release()
    
    print(f"✓ {split_name}: Processed {video_count} videos")

# Process all splits
for split in splits:
    process_split(split)

print("\n✓ All embeddings saved!")