from roboflow import Roboflow

rf = Roboflow(api_key='1b8LIsZsWgXkHtqiZr1b')

workspace = rf.workspace('my-workspace')
project = rf.project("multimodeldetection")

workspace.upload_dataset(
    "C:/Users/hp333/Desktop/Multimodel_emotion_detection/meld_frames",
    "multimodeldetection",
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    is_prediction = False
)

# import os
# import shutil

# SRC_DIR = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/meld_all_frames"
# DST_DIR = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/meld_frames"

# os.makedirs(DST_DIR, exist_ok=True)

# IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# for root, _, files in os.walk(SRC_DIR):
#     folder_name = os.path.basename(root)

#     for file in files:
#         if file.lower().endswith(IMAGE_EXTS):
#             src_path = os.path.join(root, file)

#             new_name = f"{folder_name}_{file}"
#             dst_path = os.path.join(DST_DIR, new_name)

#             shutil.move(src_path, dst_path)

# print("All images moved successfully.")
