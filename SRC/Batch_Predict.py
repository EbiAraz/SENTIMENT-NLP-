import argparse
from pathlib import Path
import csv

from Inference import predict_sentiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch sentiment prediction from CSV.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--output",
        default="Outputs/predictions.csv",
        help="Path to output CSV file with predictions.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the text column in the input CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")
        if args.text_column not in reader.fieldnames:
            raise ValueError(
                f"Text column '{args.text_column}' not found. Available columns: {reader.fieldnames}"
            )

        for row in reader:
            text = (row.get(args.text_column) or "").strip()
            row["prediction"] = predict_sentiment(text) if text else ""
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else [args.text_column, "prediction"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
