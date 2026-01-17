import torch
from Model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model.FusionClassifier().to(device)
model.load_state_dict(torch.load("Multimodel_emotion_detection/Model/checkpoints/model1.pt", map_location = device))
model.eval()

def infrence_mode_static(input_tensor: torch.Tensor):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probs = torch.softmax(output, dim = 1)
        preds = torch.argmax(probs, dim = 1)
        return preds.cpu().tolist(), probs.cpu().tolist()

def infrence_mode_dynamic(input_tensor: torch.Tensor):
    pass