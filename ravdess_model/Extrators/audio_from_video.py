import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import subprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_AUDIO_LEN = 16000 * 3  # 3 seconds

audio_model_name = "facebook/wav2vec2-base-960h"

audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name).to(DEVICE)

base_path = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/ravdess_model/data"
splits = ["Train", "Test", "Dev"]

print(f"Using device: {DEVICE}")

def extract_audio_from_mp4(mp4_path):
    """Extract audio from MP4 file using ffmpeg"""
    try:
        # Create temporary wav file
        temp_wav = mp4_path.replace('.mp4', '_temp.wav')
        
        # Use ffmpeg to extract audio
        command = [
            'ffmpeg',
            '-i', mp4_path,
            '-q:a', '9',  # Quality
            '-n',  # Don't overwrite
            temp_wav
        ]
        
        subprocess.run(command, capture_output=True, check=True)
        
        # Load audio with librosa
        audio, sr = librosa.load(temp_wav, sr=16000)
        
        # Clean up temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        return audio
    
    except Exception as e:
        print(f"Error extracting audio from {mp4_path}: {e}")
        return None

def pad_or_truncate_np(audio_array, target_len):
    """Ensure audio array is exactly target_len long"""
    if len(audio_array) > target_len:
        return audio_array[:target_len]
    elif len(audio_array) < target_len:
        return np.pad(audio_array, (0, target_len - len(audio_array)))
    return audio_array

def generate_audio_embedding(audio):
    """Generate audio embedding from audio array"""
    if audio is None or len(audio) == 0:
        return np.zeros(768, dtype=np.float32)
    
    # Pad/truncate audio
    audio = pad_or_truncate_np(audio, MAX_AUDIO_LEN)
    
    # Extract features
    audio_inputs = audio_extractor(
        [audio], sampling_rate=16000, return_tensors="pt", padding=True
    )
    audio_inputs = {k: v.to(DEVICE) for k, v in audio_inputs.items()}
    
    # Get embedding
    with torch.no_grad():
        outputs = audio_encoder(**audio_inputs)
        feature = outputs.last_hidden_state.mean(dim=1)
        feature = feature.squeeze(0).cpu().numpy().astype(np.float32)
    
    return feature

def process_split(split_name):
    """Process MP4 files and extract audio embeddings"""
    input_dir = os.path.join(base_path, split_name)
    output_dir = os.path.join(base_path, split_name, "embeddings", "Audio")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {split_name} data...")
    
    video_count = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if not file.endswith('.mp4'):
                continue
            
            # Only process files that start with modality code 01 (audio_video)
            if not file.startswith('01-'):
                continue
            
            file_path = os.path.join(root, file)
            
            try:
                # Extract audio from MP4
                audio = extract_audio_from_mp4(file_path)
                
                if audio is None:
                    print(f"⚠️  Failed to extract audio from {file}")
                    continue
                
                # Generate embedding
                embedding = generate_audio_embedding(audio)
                
                # Save embedding
                output_filename = file.replace('.mp4', '.npy')
                np.save(os.path.join(output_dir, output_filename), embedding)
                
                video_count += 1
            
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    print(f"✓ {split_name}: Extracted audio from {video_count} videos")

# Process all splits
for split in splits:
    process_split(split)

print("\n✓ All audio embeddings extracted from MP4 files!")