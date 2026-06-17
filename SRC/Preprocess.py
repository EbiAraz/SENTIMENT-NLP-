from transformers import AutoTokenizer
from Config import MODEL_CHECKPOINT, MAX_LENGTH

tokenizer  = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def TokenizeFunction(examples):
    return tokenizer(examples["text"],
                        truncation=True, 
                        padding="max_length", 
                        max_length=MAX_LENGTH
                        )