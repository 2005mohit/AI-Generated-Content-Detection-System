
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform (same as training) ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def build_model(neurons: int):
    """Exact same architecture used during training."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3 if neurons == 256 else 0.4),
        nn.Linear(in_features, neurons),
        nn.ReLU(),
        nn.Dropout(p=0.2 if neurons == 256 else 0.3),
        nn.Linear(neurons, 2)
    )
    return model.to(DEVICE)


def load_image_model(model_path: str = "model/image_model/image_model.pth"):
    """Load EfficientNet model — auto-detects hidden layer size from weights."""
    state_dict = torch.load(model_path, map_location=DEVICE)
    neurons = state_dict["classifier.1.weight"].shape[0]  # auto-detect
    model = build_model(neurons)
    model.load_state_dict(state_dict)
    model.eval()
    print(f" Image model loaded! (neurons={neurons})")
    return model


def predict_image(image: Image.Image, model) -> dict:
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    real_prob  = float(probs[0])
    ai_prob    = float(probs[1])
    confidence = float(np.max(probs)) * 100
    label      = "AI Generated" if ai_prob > 0.5 else "Real"

    return {
        "label"           : label,
        "score"           : round(ai_prob * 100, 2),
        "confidence"      : round(confidence, 2),
        "ai_probability"  : round(ai_prob, 4),
        "real_probability": round(real_prob, 4),
    }
