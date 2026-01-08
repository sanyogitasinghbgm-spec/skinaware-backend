import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

class SkinCNN:
    def __init__(self):
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights = weights)
        self.model.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        self.model.eval()
        self.head.eval()
    
    def infer(self, image_tensor):
        torch.set_grad_enabled(False)
        features = self.model(image_tensor)
        outputs = self.head(features)
        torch.set_grad_enabled(True)

        dryness = float(outputs[0][0])
        texture = float(outputs[0][1])
        redness = float(outputs[0][2])
        pigmentation = float(outputs[0][3])

        return dryness, texture, redness, pigmentation