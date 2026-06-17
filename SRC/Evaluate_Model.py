import evaluate
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, Trainer
from torch.utils.data import DataLoader
import numpy as np
from Data_Loader import load_imdb_data
from Preprocess import TokenizeFunction
from Config import MODEL_SAVE_PATH


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {
        "accuracy": accuracy.get("accuracy") if accuracy else 0,
        "f1": f1.get("f1") if f1 else 0
        }

def main():
    _, test_dataset = load_imdb_data()
    tokenized_test = test_dataset.map(TokenizeFunction, batched=True)
    tokenized_test.set_format("torch")

    model_dir = Path(MODEL_SAVE_PATH)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Train first with Train.py")

    required_files = ["config.json", "model.safetensors"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {model_dir}: {', '.join(missing)}. Run Train.py to generate them."
        )

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    from transformers import TrainingArguments
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(results_dir),
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

    results = trainer.evaluate(
        eval_dataset=tokenized_test,
        metric_key_prefix="test"
    )

    print(results)

    test_loader = DataLoader(tokenized_test, batch_size=8)

    all_logits = []
    all_labels = []

    for batch in test_loader:
        inputs = {key: val.to(model.device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
        labels = batch["label"].to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics((all_logits, all_labels))
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()