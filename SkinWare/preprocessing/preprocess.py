# Image preprocessing before feeding to the AI
import cv2
import torch

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (244, 244))
    image = image / 225.0
    image = image - 0.5
    image = image / 0.5
    tensor = torch.tensor(image, dtype = torch.float32)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    return tensor
    