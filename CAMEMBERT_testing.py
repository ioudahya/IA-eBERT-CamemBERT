import os
import json
import time
import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import TensorDataset, random_split, \
							DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, \
						 AdamW, get_linear_schedule_with_warmup
import seaborn

model = CamembertForSequenceClassification.from_pretrained(
	'camembert-base',
	num_labels = 2)

model.load_state_dict(torch.load('model.pt'))

dataset = pd.read_csv('DATASET_CAMEMBERT/file.csv')
print(dataset.head())                         


reviews = dataset['text'].values.tolist()
sentiments = dataset['label'].values.tolist()

TOKENIZER = CamembertTokenizer.from_pretrained(
    'camembert-base',
    do_lower_case=True)

def preprocess(raw_reviews, sentiments=None):
    encoded_batch = TOKENIZER.batch_encode_plus(raw_reviews,
                                                add_special_tokens=False,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors = 'pt')
    if sentiments:
        sentiments = torch.tensor(sentiments)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], sentiments
    return encoded_batch['input_ids'], encoded_batch['attention_mask']

def predict(reviews, model=model):
    with torch.no_grad():
        model.eval()
        input_ids, attention_mask = preprocess(reviews)
        retour = model(input_ids, attention_mask=attention_mask)
        print("retour",retour)
        return torch.argmax(retour[0], dim=1)




def evaluate(reviews, sentiments, metric='report'):
    predictions = predict(reviews)
    if metric == 'report':
        return metrics.classification_report(sentiments, predictions, zero_division=0)
    elif metric == 'matrix':
        return metrics.confusion_matrix(sentiments, predictions)        


split_border = int(len(sentiments)*0.8)
reviews_train, reviews_validation = reviews[:split_border], reviews[split_border:]
sentiments_train, sentiments_validation = sentiments[:split_border], sentiments[split_border:]


print(predict(['ce film est nul']))    

print(predict(['ce film est trop cool']))  

print(predict(['malgrer le bruit dans le cinéma, le film était bien']))

print(predict(['les films de denzel washington sont une tuerie ! , les films sont toujours aussi bon']))

# https://keisan.casio.com/exec/system/15168444286206 pour calculer avec softmax