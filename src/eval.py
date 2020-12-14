"""
Contains the functions used for evaluation.
Can be run as a standalone script.
"""

import config
import torch
import time
from model import BERTBaseUncased
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

MODEL = None
DEVICE = config.DEVICE
PREDICTION_DICT = dict()

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

def predict(sentence):
    start_time = time.time()
    sarcasm_prediction = sentence_prediction(sentence)
    notsarcasm_prediction = 1 - sarcasm_prediction
    response = {}
    response["response"] = {
        "sarcasm": str(sarcasm_prediction),
        "not_sarcasm": str(notsarcasm_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return response

MODEL = BERTBaseUncased()
MODEL.load_state_dict(torch.load(config.MODEL_PATH))
MODEL.to(DEVICE)
MODEL.eval()

if __name__ == "__main__":
  df = pd.read_csv(config.TESTING_FILE)
  ans = []
  print(df.head())
  for idx, line in tqdm(df.iterrows()):
    x = predict(line['tweet'])
    label = 'SARCASM' if round(float(x['response']['sarcasm'])) == 1 else 'NOT_SARCASM'
    ans.append(f"twitter_{idx+1},{label}")
  np.savetxt(f'{config.OUTPUT_PATH}/answer.txt', ans, delimiter="\n", fmt="%s")
  ans = pd.read_csv(f'{config.OUTPUT_PATH}/answer.txt', header=None, usecols=[0,1], names=['id', 'label'])
  print(ans.label.value_counts())