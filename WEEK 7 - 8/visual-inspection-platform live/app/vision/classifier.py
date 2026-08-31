"""
Supervised defect classifier (`classifier_v1`).

DefectCNN's shape here is reconstructed from the state_dict keys saved in
defect_cnn.pt (features.0/4/8 = conv layers, features.1/5/9 = batch-norm,
classifier.2 = final Linear(128, 6)). A .pt checkpoint only stores weights,
not the class definition, so this class shape has to match training exactly
for load_state_dict() to succeed.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from app.config import CLASSIFIER_MODEL_PATH, CLASS_NAMES, IMG_SIZE, IN_CHANNELS


class DefectCNN(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 0
            nn.BatchNorm2d(32),                                     # 1
            nn.ReLU(inplace=True),                                  # 2
            nn.MaxPool2d(2),                                        # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # 4
            nn.BatchNorm2d(64),                                     # 5
            nn.ReLU(inplace=True),                                  # 6
            nn.MaxPool2d(2),                                        # 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # 8
            nn.BatchNorm2d(128),                                    # 9
            nn.ReLU(inplace=True),                                  # 10
            nn.MaxPool2d(2),                                        # 11
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 0
            nn.Flatten(),              # 1
            nn.Linear(128, num_classes),  # 2
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> DefectCNN:
    """Lazily load and cache the classifier on first use."""
    global _model
    if _model is None:
        model = DefectCNN()
        state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=_device)
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()
        _model = model
    return _model


def predict(image_tensor: torch.Tensor):
    """
    image_tensor: float tensor, shape (1, IN_CHANNELS, IMG_SIZE, IMG_SIZE), values in [0, 1].

    Returns a dict:
        {
          "predicted_class": str,
          "confidence": float,           # softmax prob of predicted_class
          "class_probabilities": {name: prob, ...},
          "inference_time_ms": float,
        }
    """
    model = load_model()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(image_tensor.to(_device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    class_probabilities = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    top_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "class_probabilities": class_probabilities,
        "inference_time_ms": elapsed_ms,
    }
