import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import librosa
import os
from pathlib import Path
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_AUDIO_LEN = 16000 * 3  # 3 sec clips

audio_model_name = "facebook/wav2vec2-base-960h"
audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name).to(DEVICE)

base_path = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/ravdess_model/data"
splits = ["Train", "Test", "Dev"]

def pad_or_truncate_np(audio_array, target_len):
    if len(audio_array) > target_len:
        return audio_array[:target_len]
    elif len(audio_array) < target_len:
        return np.pad(audio_array, (0, target_len - len(audio_array)))
    return audio_array

def process_split(split_name):
    input_dir = os.path.join(base_path, split_name)
    output_dir = os.path.join(base_path, split_name, "embeddings", "Audio")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {split_name} data...")
    
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if not file.endswith('.wav'):
                continue
            
            file_path = os.path.join(root, file)
            
            try:
                # Load and process audio
                audio, _ = librosa.load(file_path, sr=16000)
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
                
                # Save embedding
                output_filename = file.replace('.wav', '.npy')
                np.save(os.path.join(output_dir, output_filename), feature)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Process all splits
for split in splits:
    process_split(split)

print("\n✓ All embeddings saved!")