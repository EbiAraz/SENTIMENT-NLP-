from pathlib import Path
import subprocess
import sys
import argparse


def run_step(script_name: str, extra_args: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent
    script_path = root / "SRC" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n=== Running {script_name} ===")
    command = [sys.executable, str(script_path)]
    if extra_args:
        command.extend(extra_args)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_name} (exit code {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full sentiment pipeline.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run training in smoke-test mode for a quick pipeline check.",
    )
    args = parser.parse_args()

    train_args = ["--smoke-test"] if args.fast else None
    run_step("Train.py", train_args)
    run_step("Evaluate_Model.py")
    run_step("Inference.py")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
