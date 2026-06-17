import argparse
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Config import MODEL_CHECKPOINT, MODEL_SAVE_PATH, MAX_LENGTH

label_map = {0: "Negative", 1: "Positive"}


@lru_cache(maxsize=1)
def load_model_bundle():
    model_dir = Path(MODEL_SAVE_PATH)
    required_files = ["config.json", "model.safetensors"]
    if model_dir.exists() and all((model_dir / name).exists() for name in required_files):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT)
    model.eval()
    return tokenizer, model

def predict_sentiment(text, return_scores=False):
    text = (text or "").strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    tokenizer, model = load_model_bundle()
    inputs = tokenizer(text, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=MAX_LENGTH
                        )
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    predicted_label_id = int(torch.argmax(probabilities).item())
    label = label_map[predicted_label_id]

    if not return_scores:
        return label

    return {
        "label": label,
        "label_id": predicted_label_id,
        "confidence": float(probabilities[predicted_label_id].item()),
        "probabilities": {
            label_map[index]: float(probability.item())
            for index, probability in enumerate(probabilities)
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for a given text.")
    parser.add_argument(
        "--text",
        type=str,
        default="I really enjoyed this movie! The plot was engaging and the acting was superb.",
        help="Input text to classify.",
    )
    args = parser.parse_args()

    prediction = predict_sentiment(args.text, return_scores=True)
    print(f"Text: {args.text}\nPredicted Sentiment: {prediction['label']} ({prediction['confidence']:.2%})")