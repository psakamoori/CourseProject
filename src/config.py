"""
Contains the application configuration and common settings used.
"""

import transformers

DEVICE = "cpu"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../input/model.bin"
TRAINING_FILE = "../input/train.csv"
TRAINING_FILE_ORIGINAL = "../input/data/train.jsonl"
TESTING_FILE = "../input/test.csv"
TESTING_FILE_ORIGINAL = "../input/data/test.jsonl"
OUTPUT_PATH = "../output/"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)