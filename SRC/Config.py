from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 256
TRAIN_SIZE = 2000
TEST_SIZE = 500
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
OUTPUT_DIR = str(BASE_DIR / "Outputs" / "distilbert_imdb")
MODEL_SAVE_PATH = str(BASE_DIR / "Models" / "distilbert_imdb")