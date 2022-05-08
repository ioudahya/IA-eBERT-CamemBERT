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


split_border = int(len(sentiments)*0.8)
reviews_train, reviews_validation = reviews[:split_border], reviews[split_border:]
sentiments_train, sentiments_validation = sentiments[:split_border], sentiments[split_border:]

input_ids, attention_mask, sentiments_train = preprocess(reviews_train, sentiments_train)
# Combine the training inputs into a TensorDataset
train_dataset = TensorDataset(
	input_ids,
	attention_mask,
	sentiments_train)

input_ids, attention_mask, sentiments_validation = preprocess(reviews_validation, sentiments_validation)
# Combine the validation inputs into a TensorDataset
validation_dataset = TensorDataset(
	input_ids,
	attention_mask,
	sentiments_validation)    


# size of 16 or 32.
batch_size = 16

# Create the DataLoaders
train_dataloader = DataLoader(
			train_dataset,
			sampler = RandomSampler(train_dataset),
			batch_size = batch_size)

validation_dataloader = DataLoader(
			validation_dataset,
			sampler = SequentialSampler(validation_dataset),
			batch_size = batch_size)

model = CamembertForSequenceClassification.from_pretrained(
	'camembert-base',
	num_labels = 2).to('cuda')


optimizer = AdamW(model.parameters(),
				  lr = 2e-5, # Learning Rate
				  eps = 1e-8) # Epsilon)
epochs = 3

device = torch.device('cuda')
 

training_stats = []
 

for epoch in range(0, epochs):
	 
	print("")
	print(f'########## Epoch {epoch+1} / {epochs} ##########')
	print('Training...')
	t0 = time.time()
 
	total_train_loss = 0
 

	model.train()
 
	# Pour chaque batch
	for step, batch in enumerate(train_dataloader):
 
		if step % 40 == 0 and not step == 0:
			print(f'  Batch {step}  of {len(train_dataloader)}.')
		 

		input_id = batch[0].to(device)
		attention_mask = batch[1].to(device)
		sentiment = batch[2].to(device)
 
		model.zero_grad()        
 
		loss, logits = model(input_id, 
							 token_type_ids=None, 
							 attention_mask=attention_mask, 
							 labels=sentiment,
							 return_dict=False)


		total_train_loss += loss.item()
 
		# Backpropagtion
		loss.backward()
 
		optimizer.step()
 
	 
	training_stats.append(
		{
			'epoch': epoch + 1,
			'Training Loss': avg_train_loss,
		}
	)
 
print("Model saved!")
torch.save(model.state_dict(), "model.pt")


