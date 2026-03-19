from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Labels from Hello-SimpleAI/chatgpt-detector-roberta ─────────────────────
# Label 0 = Human Written, Label 1 = ChatGPT / AI Generated
ID2LABEL = {0: "Human Written", 1: "AI Generated"}


def load_text_model(model_path: str = "model/text_model"):
    """Load RoBERTa tokenizer and model from local folder."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_text(text: str, tokenizer, model, max_length: int = 512) -> dict:
    """
    Predict if text is AI-generated or Human written.

    Returns:
        {
            "label": "AI Generated" | "Human Written",
            "score": float (0-100),
            "confidence": float (0-100),
            "ai_probability": float (0-1),
            "human_probability": float (0-1)
        }
    """
    if not text or not text.strip():
        return {
            "label": "N/A",
            "score": 0.0,
            "confidence": 0.0,
            "ai_probability": 0.0,
            "human_probability": 0.0,
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    ai_prob = float(probs[1])
    human_prob = float(probs[0])
    confidence = float(np.max(probs)) * 100

    label = ID2LABEL[1] if ai_prob > 0.5 else ID2LABEL[0]
    score = ai_prob * 100

    return {
        "label": label,
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "ai_probability": round(ai_prob, 4),
        "human_probability": round(human_prob, 4),
    }
