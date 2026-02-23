import os
import pandas as pd

base_path = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/ravdess_model/data"
splits = ["Train"]
emotions = {
    '01': 'neutral',
    '02': 'clam',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgusted',
    '08': 'surprised'
}

def create_csv_from_files(split_name):
    split_path = os.path.join(base_path, split_name)
    data = []
    print(f"\nProcessing {split_name}")
    for root, dirs, files in os.walk(split_path):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.wav'):
                file = os.path.splitext(file)[0]
                parts = file.split('-')
                data.append({
                    'modality': parts[0],
                    'vocal_channel': parts[1],
                    'emotion_code': parts[2],
                    'intensity': parts[3],
                    'stat_id': parts[4],
                    'repetition': parts[5],
                    'actor': parts[6],
                    'emotion': emotions.get(parts[2], 'unkown'),
                    'file_name': file
                })

    df = pd.DataFrame(data)
    df.to_csv(f'ravdess_mulitmodal_{split_name}.csv', index = False)
    print(f"Created csv with {len(df)} records.")


for split in splits:
    create_csv_from_files(split)