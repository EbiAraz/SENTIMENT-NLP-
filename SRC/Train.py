import evaluate
import numpy as np
import inspect
import argparse
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from Data_Loader import load_imdb_data
from Preprocess import TokenizeFunction
from Config import (   
    MODEL_CHECKPOINT,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    OUTPUT_DIR,
    MODEL_SAVE_PATH
    )

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

def parse_args():
    parser = argparse.ArgumentParser(description="Train sentiment model.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick training smoke test on a very small subset.",
    )
    parser.add_argument(
        "--smoke-train-size",
        type=int,
        default=128,
        help="Train subset size for smoke test mode.",
    )
    parser.add_argument(
        "--smoke-test-size",
        type=int,
        default=64,
        help="Eval subset size for smoke test mode.",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=float,
        default=0.1,
        help="Epochs for smoke test mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(OUTPUT_DIR)
    model_save_dir = Path(MODEL_SAVE_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = load_imdb_data()
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Loaded empty dataset. Check TRAIN_SIZE/TEST_SIZE and dataset availability.")

    if args.smoke_test:
        train_size = min(args.smoke_train_size, len(train_dataset))
        eval_size = min(args.smoke_test_size, len(test_dataset))
        train_dataset = train_dataset.select(range(train_size))
        test_dataset = test_dataset.select(range(eval_size))
        print(f"Smoke test mode: train={train_size}, eval={eval_size}, epochs={args.smoke_epochs}")

    tokenized_train = train_dataset.map(TokenizeFunction, batched=True)
    tokenized_test = test_dataset.map(TokenizeFunction, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)

    eval_arg_name = "evaluation_strategy"
    if "evaluation_strategy" not in inspect.signature(TrainingArguments.__init__).parameters:
        eval_arg_name = "eval_strategy"

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **{eval_arg_name: "epoch"},
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.smoke_epochs if args.smoke_test else EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )


    trainer.train()
    trainer.save_model(str(model_save_dir))
    print("Model saved to:", model_save_dir)

if __name__ == "__main__":
    main()    