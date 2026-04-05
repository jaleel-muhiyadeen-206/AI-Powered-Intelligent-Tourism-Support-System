"""
BERT Model Downloader
=====================
Downloads nlptown/bert-base-multilingual-uncased-sentiment to models/bert_sentiment/
for offline use by the preprocessing pipeline.

Usage:
    export HF_TOKEN=your_huggingface_token
    python download_bert_model.py
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bert_sentiment')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bert_cache')


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Downloading {MODEL_NAME} (public model, no token needed)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    tokenizer.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)

    print(f"Saved to {LOCAL_DIR}")
    print("Files:", os.listdir(LOCAL_DIR))


if __name__ == "__main__":
    main()
