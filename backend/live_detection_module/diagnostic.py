import torch
import sys
import os

print("Starting diagnostic...")
try:
    import configs.config as config
    print("Config imported successfully.")
    print(f"Device: {config.DEVICE}")
    print(f"Fusion model path: {config.FUSION_MODEL}")
    
    if os.path.exists(config.FUSION_MODEL):
        print(f"Fusion model file found at {config.FUSION_MODEL}")
    else:
        print(f"CRITICAL: Fusion model file NOT FOUND at {config.FUSION_MODEL}")
        print(f"Checking models directory: {os.listdir('models')}")
        print(f"Checking fusion_model directory: {os.listdir('models/fusion_model')}")

    from models import Model
    print("Models imported successfully.")
    
    model = Model.FusionClassifier().to(device=config.DEVICE)
    print("FusionClassifier initialized.")
    
    checkpoint = torch.load(config.FUSION_MODEL, map_location=config.DEVICE)
    print("Checkpoint loaded.")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print("State dict loaded.")
    
    print("SUCCESS: All models and checkpoints loaded correctly.")
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
