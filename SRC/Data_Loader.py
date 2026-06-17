from pathlib import Path

from datasets import load_dataset, load_from_disk
from Config import TRAIN_SIZE, TEST_SIZE


def load_imdb_data():
    local_dataset_path = Path(__file__).resolve().parent.parent / "Data" / "imdb_hf"

    if local_dataset_path.exists():
        dataset = load_from_disk(str(local_dataset_path))
    else:
        dataset = load_dataset("imdb")
        local_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(local_dataset_path))

    train_size = min(TRAIN_SIZE, len(dataset["train"]))
    test_size = min(TEST_SIZE, len(dataset["test"]))

    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(test_size))
    return train_dataset, test_dataset