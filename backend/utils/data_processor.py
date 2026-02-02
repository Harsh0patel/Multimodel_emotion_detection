import numpy as np
import cv2
from typing import List

class DataProcessor:
    """Handles decoding of raw data"""
    def __init__(self, target_size = (224, 224)):
        self.target_size = target_size
    
    def decode_frames(self, frame_bytes_list: List[bytes]) -> np.ndarray:
        """Decode JPEG bytes to numpy frames"""
        frames = []
        for frame_bytes in frame_bytes_list:
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
        
        return np.array(frames) if frames else np.array([])
    
    def decode_audio(self, audio_bytes_list: List[bytes]) -> np.ndarray:
        """Decode audio bytes to numpy array"""
        # Combine all audio chunks
        audio_data = b''.join(audio_bytes_list)
        
        # Convert to int16 array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize to float32 [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array