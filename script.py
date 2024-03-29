# -*- coding: utf-8 -*-

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
MAX_EPOCHS=10
"""Tweet-Intimacy-Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/aishafarooque/Tweet-Intimacy-Analysis/blob/main/Tweet-Intimacy-Analysis.ipynb

# Multilingual Tweet Intimacy Analysis

## Introduction

Dataset created by: Jiaxin Pei, Francesco Barbieri, Vítor Silva, Maarten Bos, Yozen Liu, Leonardo Neves, David Jurgens

The goal of this project is to train machine learning (ML) models to recognize "intimacy" in text communications. The authors of this paper define intimacy as "closeness and interdependence, self-disclosure, and warmth or affection” expressed in the language used to communicate.

We have used two datasets, Reddit questions and Multilingual tweets, in this notebook. They have both been compiled by Pei et al. and are used to study if knowledge about intimacy levels of text communication can be transferred easily from tweets to questions or vice versa.

Install dependencies
"""

# ! pip install --upgrade pip
# ! pip install transformers
# ! pip install tqdm

"""## Data

Downloading the Twitter dataset from the Multilingual Tweet Intimacy Analysis Codalab competition ([source](https://codalab.lisn.upsaclay.fr/competitions/7096#learn_the_details-overview)).
"""

# Remove Twitter's train.csv if it already exists
# ! rm -rf train.csv

# Download Twitter's training data
# ! wget -P /content https://raw.githubusercontent.com/aishafarooque/Tweet-Intimacy-Analysis/main/train.csv

# Rename train.csv -> twitter_train.csv for more clarity
# ! mv /content/train.csv /content/twitter_train.csv

"""Downloading the Reddit dataset from the author's GitHub repository: Quantifying-Intimacy-in-Language
 ([source](https://github.com/Jiaxin-Pei/Quantifying-Intimacy-in-Language/blob/main/data/annotated_question_intimacy_data.zip)).
"""

# Sanitize working directory
# ! rm -rf /content/__MACOSX
# ! rm -rf /content/annotated_question_intimacy_data

# Removing the .zip file if it already exists.
# ! rm -rf /content/annotated_question_intimacy_data.zip

# Download the dataset from the author's GitHub repository.
# ! wget -P /content https://raw.githubusercontent.com/Jiaxin%2DPei/Quantifying%2DIntimacy%2Din%2DLanguage/main/data/annotated_question_intimacy_data.zip

# Unzip the file. 
# ! unzip /content/annotated_question_intimacy_data.zip -d /content/

"""## Data fact study

### Twitter Dataset

The Twitter dataset has a total of 9491 rows and 3 columns which are:
* Tweet - Textual content of the tweet
* Intimacy Label - Intimacy score of the tweet, ranging from 1 (least intimate) to 5 (most intimate).
* Language - The language the tweet is written in. There are six languages in this datasset:  English, Spanish, Italian, Portuguese, French, and Chinese.
"""

import pandas as pd

twitter_df_train = pd.read_csv('twitter_train.csv', on_bad_lines='skip')
twitter_df_train = twitter_df_train.rename(columns={'text': 'document', 'label': 'label'})

print("Dataset size:", len(twitter_df_train), '\n')
twitter_df_train.info()

twitter_df_train.sample(5)

"""Looking at the distribution of tweets in the dataset we can see that, approximately, there are equal number of tweets across all six languages."""

tweet_distribution = twitter_df_train.groupby('language').count()['document'] \
	.reset_index().sort_values(by='document', ascending=False)
tweet_distribution.style.background_gradient()

"""### Reddit Dataset

The Reddit dataset has a total of 1797 rows and 2 columns which are:
* Question - Textual content of the Reddit question in English
* Intimacy Score - Intimacy score of the tweet, ranging from -1 (least intimate) to 1 (most intimate).
"""

reddit_df_train = pd.read_csv('final_train.txt',
							  sep='\t', header=None, names=['document', 'label'])

print("Dataset size:", len(reddit_df_train), '\n')
reddit_df_train.info()

reddit_df_train.head()

"""#### Performing Linear Mapping on the Reddit Dataset

Since the ```intimacy_scores``` in the Reddit dataset is on a range from -1 to 1, we will linearly map them from 1 to 5. The linear mapping with maintain a constant ratio between the points. 

We will perform the following:
- A scaling operation to adjust the ranges to the same size, and
- An offset operation to adjust range alignment. 

Source: http://learnwebgl.brown37.net/08_projections/projections_mapping.html 
"""

A, B, C, D = -1, 1, 1, 5
scale = (D - C) / (B - A)
offset = -A * (D - C) / (B - A) + C

for index, row in reddit_df_train.iterrows():
	iScore = row['label']

	# If the cell is re-run without clearing local variables, we'll
	# double convert the values between the 1-5 range resulting in values between
	# 5-10. This condition makes sure original scores from Reddit are not already
	#  greater than 1.
	if iScore > 1:
		break

	q = iScore * scale + offset
	reddit_df_train.at[index, 'label'] = round(q, 1)

reddit_df_train.head()

"""## Define the Tokenizer"""

# ! pip
# install
# sentencepiece
# ! pip
# install
# transformers
# ! pip
# install
# seqeval
# ! pip
# install
# bertviz
# ! pip
# install
# datasets

import datasets
import transformers

# set up verbosity of libraries
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

import torch

# Set up device. Recommend to use GPU to accelerate training
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PRETRAINED_MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

IGNORE_INDEX = -100

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
tokenizer

"""## Data split

The data will be split into training and testing datasets.
The training dataset will be 80% of the data from the Reddit and Twitter datasets. This dataset will be used to fine tune the model. 
"""

# Declare constants for commonly used strings
TWEET = 'tweet'
REDDIT = 'reddit'
TRAIN = 'train'
TEST = 'test'
COMBINED = 'combined'


def split_dataset(dataset, train_size, test_size):
	X = dataset['document']
	y = dataset['label']

	# Bin size = 1.0
	bins = np.linspace(start=1.0, stop=5.0, num=5)

	binned_y = np.digitize(y, bins)
	X_train, X_rem, y_train, y_rem = train_test_split(X, y,
													  stratify=binned_y,
													  train_size=train_size)

	binned_y_rem = np.digitize(y_rem, bins)
	X_test, X_valid, y_test, y_valid = train_test_split(X_rem, y_rem,
														stratify=binned_y_rem,
														test_size=test_size)

	print(f'X_train: {X_train.shape}, X_rem: {X_rem.shape}, y_train: {y_train.shape}, y_rem: {y_rem.shape}')
	print(f'X_test: {X_test.shape}, X_valid: {X_valid.shape}, y_test: {y_test.shape}, y_valid: {y_valid.shape}')

	return {'train': {'X': X_train, 'y': y_train},
			'test': {'X': X_test, 'y': y_test},
			'valid': {'X': X_valid, 'y': y_valid}}


"""
Combines collections from the same split of two datasets into one collection to use as a split of the combined dataset.
"""


def combine_sets(a, b):
	# Simple concatenation. Is there another way we want to try combining them
	#  (e.g. zipping/mixing them together?)
	return [a_item for a_item in a] + [b_item for b_item in b]


def combine_splits(a_split, b_split):
	return {split_name: {variable: combine_sets(a_split[split_name][variable], b_split[split_name][variable])
						 for variable in a_split[split_name]}
			for split_name in a_split}


from sklearn.model_selection import train_test_split
import numpy as np

# Pull x and y values from dataframe
datasets = {'tweet': twitter_df_train, 'reddit': reddit_df_train}

# Define train:test:valid split ratios
split_ratios = {'tweet': {'train_total_ratio': 0.8,
						  'test_rem_ratio': 0.5},

				# Train:test:valid ratio of 8:1:1 for Reddit dataset
				'reddit': {'train_total_ratio': 0.8,
						   'test_rem_ratio': 0.5}}

# Split individual tweet/reddit datasets
split_datasets = {key: split_dataset(datasets[key],
									 split_ratios[key]['train_total_ratio'],
									 split_ratios[key]['test_rem_ratio'])
				  for key in datasets}

# Create and split combined dataset
split_datasets['combined'] = combine_splits(split_datasets['tweet'],
											split_datasets['reddit'])

split_datasets.keys()

print(f'Keys in {TWEET} are {split_datasets[TWEET].keys()}')
print(f'Keys in {REDDIT} are {split_datasets[REDDIT].keys()}')
print(f'Keys in {COMBINED} are {split_datasets[COMBINED].keys()}')

# Ensure the features were correctly combines
features_in_twitter = len(split_datasets[TWEET][TRAIN]['X'])
features_in_reddit = len(split_datasets[REDDIT][TRAIN]['X'])
features_in_combined = len(split_datasets[COMBINED][TRAIN]['X'])

assert features_in_combined == (features_in_twitter + features_in_reddit), f'Features do not match'

# Ensure the labels were correctly combined
labels_in_twitter = len(split_datasets[TWEET][TRAIN]['y'])
labels_in_reddit = len(split_datasets[REDDIT][TRAIN]['y'])
labels_in_combined = len(split_datasets[COMBINED][TRAIN]['y'])

assert labels_in_combined == (labels_in_twitter + labels_in_reddit), f'Labels do not match'

"""After splitting the dataset, we will be left with the following structure:

```
split_datasets
├── tweet
│   ├── train
│   │   ├── X
│   │   └── y
│   ├── test
│   └── valid
├── reddit
│   ├── train
│   ├── test
│   └── valid
└── combined
    ├── train
    ├── test
    └── valid
```

## Research regarding data and data split

Pei and Jurgens (2020) conducted analyses for an issue in pre-processing, where they were only able to do it for 1,000 tweets. They found that the distribution of the final annotated intimacy scores are not changed much while the fine-tuned XLM-T only achieved a Pearson’s r of 0.43 on the random sample, suggesting that the model trained on Reddit questions may not be reliable enough to detect intimacy in tweets. A potential goal of ours is too improve upon this making the model reliable for even reddit questions.

(Liu et al., 2019) include two variants: one which is fine-tuned on 3M unannotated questions on a masked language modeling task, and a second which uses the default parameters in RoBERTa. Training uses only the 2,247 annotated Reddit questions, split 8:1:1 into training, validation, and test.

## Preparing data

Before sending the data to the model, we will preprocess it.

### Create save location for models
"""

# Attach to the Google drive persistent storage
# from google.colab import drive
#
# drive.mount('/content/drive')

# Create the folder for saving models if it doesn't already exist
import os
from os import path
#
# os.chdir('/content/drive/MyDrive')
if not path.exists('download_models'):
	os.mkdir('download_models')

"""### Define the datasets

We will have identifiers for each word in a language produced by tokenization. During this process, chunks of words, such as a sentence or a phrase, are broken down into smaller units. Each unit is called a token which could be words, numbers, or punctuation marks. 

Any word not in our dictionary will be replaced with the `unknown` token. 

Reference:
- [Tokenizer - huggingface.co](https://huggingface.co/docs/transformers/main_classes/tokenizer)
"""


def tokenize_catch_unknown(word, tokenizer, vocab):
	"""
	Description: Tokenizes a word if it is known, but catches unknown words and replaces it with unknown tokens.

	"""
	if vocab.get(word):
		return tokenizer.tokenize(word)
	else:
		return [tokenizer.unk_token]


def prepare_document_encoding(document, tokenizer, vocab):
	"""
	Description: Converts one document (a tweet or other text) into tokens in the model's vocabulary.
	Returns a dictionary containing the tokens as text, the numerical ids of the tokens, and the score label for the document (already numerical here)

	"""
	tokenized = [tokenizer.cls_token] + [token for word in document for token in
										 tokenize_catch_unknown(word, tokenizer, vocab)]
	input_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenized]

	return {"tokenized": tokenized,
			"input_ids": input_ids}


def prepare_data(data, tokenizer, vocab):
	"""
	Description: Converts the documents in the data into tokens in the model's vocabulary.

	"""
	for split in data:
		data[split]['tokenized'] = []
		data[split]['input_ids'] = []
		data[split]['labels'] = []

		for document, label in zip(data[split]['X'], data[split]['y']):
			prepared_document = prepare_document_encoding(document, tokenizer, vocab)

			data[split]['tokenized'] += [prepared_document['tokenized']]
			data[split]['input_ids'] += [prepared_document['input_ids']]
			data[split]['labels'] += [label]

	return data


import torch
from torch.utils.data import Dataset


class IADataset(Dataset):
	def __init__(self, data, pad_token_id):
		self.tokenized = data['tokenized']
		self.input_ids = data['input_ids']
		self.labels = data['labels']

		self.pad_token_id = pad_token_id

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, index):
		return self.input_ids[index], self.labels[index]

	# Try without batch collation
	def collate_fn(self, batch):
		"""
		Description:
			This function collates lists of samples into batches. It should be passed as the `collate_fn` argument when creating dataloaders.
		Inputs:
			- batch (List[Tuple]): a list of tuples. The tuple (Tuple[List]) in the batch is the return value (List[List]) of
			  `__getitem(self, index)` function. The elements are: 1) a List[int] (input_ids) and 2) a real (label).
		Outputs:
			- input_dict (Dict[str, torch.LongTensor]): a dictionary containing a mapping between input names and input values. The `input_ids`
			  (torch.LongTensor of shape (batch_size, sequence_length)) and `labels` (torch.FloatTensor of shape (batch_size, sequence_length))
			  in the dictionary are token indexes and label values, respectively.
		"""

		# unwrap the batch into every field
		input_ids, labels = map(list, zip(*batch))

		max_length = max(map(len, input_ids))

		padded_input_ids = [i + [self.pad_token_id] * (max_length - len(i)) for i in input_ids]
		attention_masks = [[1 for id in i] + [0] * (max_length - len(i)) for i in input_ids]

		input_dict = {
			'input_ids': torch.tensor(padded_input_ids).long(),
			'attention_masks': torch.tensor(attention_masks).long(),
			'labels': torch.tensor(labels).to(torch.float64)
		}

		return input_dict


def get_ia_splits(data, pad_token_id):
	"""
	Description: Creates IADatasets around each split of the dataset.
	"""
	ia_splits = {split: IADataset(data[split], pad_token_id)
				 for split in data}

	return ia_splits


# Convert the documents into sequences of input ids in the model's vocabulary
vocab = tokenizer.get_vocab()

prepared_datasets = {key: prepare_data(split_datasets[key], tokenizer, vocab) for key in split_datasets}

prepared_datasets.keys()

assert len(prepared_datasets[TWEET][TRAIN][
			   'X']) == features_in_twitter, f"Mismatched sizes, expected {features_in_twitter}, got {len(prepared_datasets[TWEET][TRAIN]['X'])}"
assert len(prepared_datasets[REDDIT][TRAIN][
			   'X']) == features_in_reddit, f"Mismatched sizes, expected {features_in_reddit}, got {len(prepared_datasets[REDDIT][TRAIN]['X'])}"
assert len(prepared_datasets[COMBINED][TRAIN][
			   'X']) == features_in_combined, f"Mismatched sizes, expected {features_in_combined}, got {len(prepared_datasets[COMBINED][TRAIN]['X'])}"

assert len(prepared_datasets[TWEET][TRAIN][
			   'y']) == labels_in_twitter, f"Mismatched sizes, expected {labels_in_twitter}, got {len(prepared_datasets[TWEET][TRAIN]['y'])}"
assert len(prepared_datasets[REDDIT][TRAIN][
			   'y']) == labels_in_reddit, f"Mismatched sizes, expected {labels_in_reddit}, got {len(prepared_datasets[REDDIT][TRAIN]['y'])}"
assert len(prepared_datasets[COMBINED][TRAIN][
			   'y']) == labels_in_combined, f"Mismatched sizes, expected {labels_in_combined}, got {len(prepared_datasets[COMBINED][TRAIN]['y'])}"

ia_datasets = {key: get_ia_splits(prepared_datasets[key],
								  pad_token_id=tokenizer.pad_token_id)
			   for key in prepared_datasets}
ia_datasets

assert len(ia_datasets[TWEET][
			   TRAIN].tokenized) == features_in_twitter, f"Mismatched sizes, expected {features_in_twitter}, got {len(ia_datasets[TWEET][TRAIN].tokenized)}"
assert len(ia_datasets[REDDIT][
			   TRAIN].tokenized) == features_in_reddit, f"Mismatched sizes, expected {features_in_reddit}, got {len(ia_datasets[REDDIT][TRAIN].tokenized)}"
assert len(ia_datasets[COMBINED][
			   TRAIN].tokenized) == features_in_combined, f"Mismatched sizes, expected {features_in_combined}, got {len(ia_datasets[COMBINED][TRAIN].tokenized)}"

print('--TRAIN--')
print(f'IA Dataset Twitter length: {len(ia_datasets[TWEET][TRAIN].tokenized)}')
print(f'IA Dataset Reddit length: {len(ia_datasets[REDDIT][TRAIN].tokenized)}')
print(f'IA Dataset combined length: {len(ia_datasets[COMBINED][TRAIN].tokenized)}')

print('--TEST--')
print(f'IA Dataset Twitter length: {len(ia_datasets[TWEET][TEST].tokenized)}')
print(f'IA Dataset Reddit length: {len(ia_datasets[REDDIT][TEST].tokenized)}')
print(f'IA Dataset combined length: {len(ia_datasets[COMBINED][TEST].tokenized)}')

assert len(ia_datasets[TWEET][
			   TRAIN].tokenized) == features_in_twitter, f"Mismatched sizes, expected {features_in_twitter}, got {len(ia_datasets[TWEET][TRAIN].tokenized)}"
assert len(ia_datasets[REDDIT][
			   TRAIN].tokenized) == features_in_reddit, f"Mismatched sizes, expected {features_in_reddit}, got {len(ia_datasets[REDDIT][TRAIN].tokenized)}"
assert len(ia_datasets[COMBINED][
			   TRAIN].tokenized) == features_in_combined, f"Mismatched sizes, expected {features_in_combined}, got {len(ia_datasets[COMBINED][TRAIN].tokenized)}"

"""The folder structure for `ia_datasets` will be:
```
ia_datasets
├── tweet
│   ├── train
│   │   ├── tokenized
│   │   ├── input_ids
│   │   ├── labels
│   │   └── pad_token_ids
│   ├── test
│   └── valid
├── reddit
└── combined
```
"""

# Cleanup to save RAM
del prepared_datasets
del split_datasets
del reddit_df_train
del twitter_df_train

import gc

gc.collect()

"""### Define the DataLoaders

Dataloaders are constructs of the PyTorch library which define and control data preprocessing. The data will be fed in batches to the model. This is important because it is inefficient to load the data altogether in memory at once. The size of the data loaded at any given time into memory is controlled by the `batch_size` which is 32 for the training, validation and testing datasets.  

Every DataLoader has a Sampler which is used internally to get the indices for each batch. 

The `SequentialSampler` iterates over the dataset in a sequential order. For example: `[1,2,3]` -> `1,2,3`. Here the `shuffle` parameter is set to `false`.

The `RandomSampler` is just like it's sequential counterpart, but with `shuffle=True`.


References:
- [torch.utils.data — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/data.html)
- [SequentialSampler](https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#SequentialSampler)
"""

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler


def get_dataloaders(datasets, train_batch_size, eval_batch_size):
	"""
	Description:
		This function implements batch training by creating dataloaders for datasets to acclerate training.
	Inputs:
		- datasets (Dict[str, Dict]): a dictionary containing a mapping between dataset names and dataset values.
		- train_batch_size (int): an integer which is used as the batch size when creating the train dataloader
		- eval_batch_size (int): an integer which is used as the batch size when creating the validation and test dataloader
	"""
	# Retrieve the dataset split labels ("train", "validation", "test")
	splits = datasets.keys()

	# Choose different parameters for each dataset split
	sampler_classes = {split:
						   (RandomSampler if split == "train" else SequentialSampler)
					   for split in splits}

	batch_sizes = {split:
					   (train_batch_size if split == "train" else eval_batch_size)
				   for split in splits}

	# Initialize the BatchSamplers with the differentiated parameters
	batch_samplers = {split:
						  BatchSampler(sampler_classes[split](datasets[split]),
									   batch_sizes[split],
									   drop_last=False)
					  for split in splits}

	# Build the dataloaders from the initialized BatchSamplers and the custom collate_fn
	dataloaders = {split:
					   DataLoader(datasets[split],
								  batch_sampler=batch_samplers[split],
								  collate_fn=datasets[split].collate_fn)
				   for split in splits}

	return dataloaders




dataloaders = {
	key: get_dataloaders(ia_datasets[key], train_batch_size=TRAIN_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE)
	for key in ia_datasets}
dataloaders

"""Check to ensure that lengths of dataset are as expected."""

import math


def check_size(dataType, partitionType):
	if partitionType == TRAIN:
		batchSize = TRAIN_BATCH_SIZE
	else:
		batchSize = EVAL_BATCH_SIZE

	got = len(dataloaders[dataType][partitionType])
	expected = math.ceil(len(ia_datasets[dataType][partitionType].tokenized) / batchSize)
	assert got == expected, f'Got {got}, expected {expected}'

	print(f'{dataType}\'s {partitionType} length is {expected}.')


for d in [TWEET, REDDIT, COMBINED]:
	for p in [TRAIN, TEST, 'valid']:
		check_size(d, p)
	print(f'{d} is good.\n')

print('All good - sizes are as expected!')

"""## RoBERTa Model Architecture"""

from transformers import AutoConfig

# set up the configuration for BERT model
config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)
config

"""#### RobERTa Regressor - New"""

import torch

PRETRAINED_MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE

import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel


class XLMRobertaRegressor(nn.Module):

	def __init__(self, pretrained_model_name_or_path, drop_rate=0.2, freeze_camembert=False):
		super(XLMRobertaRegressor, self).__init__()
		D_in, D_out = 768, 1

		self.xlmroberta = XLMRobertaModel.from_pretrained(pretrained_model_name_or_path)
		self.regressor = nn.Sequential(
			nn.Dropout(drop_rate),
			nn.Linear(D_in, D_out)
		)

	def forward(self, input_ids, attention_masks):
		outputs = self.xlmroberta(input_ids, attention_masks)
		class_label_output = outputs[1]
		outputs = self.regressor(class_label_output)
		return outputs


"""## Training

In the report (Liu et al., 2019), the parameters: 
- Batch size is 128 and learning rate is 0.0001 with max length set to 50.
- AdamW (Kingma and Ba, 2014) used for optimization.
- All the other hyperparameters and the model size are the same as the default roberta-base model

**Training**
They trained model for five epochs, selecting model with lowest MSE on validation set. 

**fine-tuning process**: They followed all default settings recommended by Hugging Face.

**Tuning Learning Rate**: 0.0001 and 0.00001 both achieved good scores regarding MSE and Pearson r.

#### Training Loop

Define the custom training loop for the regression task.
"""

from torch.nn.utils.clip_grad import clip_grad_norm
from tqdm import tqdm


def train(model, optimizer, scheduler, loss_function, epochs,
		  train_dataloader, device, clip_value=2):
	for epoch in range(epochs):
		print(f'Epoch #{epoch}')

		best_loss = 1e10
		model.train()

		for step, batch in enumerate(tqdm(train_dataloader)):
			batch_inputs, batch_masks, batch_labels = \
				tuple(batch[field].to(device) for field in batch)

			model.zero_grad()
			outputs = model(batch_inputs, batch_masks)

			outputs = outputs.to(torch.float64)
			batch_masks = batch_masks.to(torch.float64)

			loss = loss_function(outputs.squeeze(),
								 batch_labels.squeeze())

			loss.backward()
			clip_grad_norm(model.parameters(), clip_value)
			optimizer.step()
			scheduler.step()

	return model


"""Train on one dataset, save to persistent storage and delete to free up space."""

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

SAVED_MODEL_PATH = 'download_models/model_state_dict_'


def train_and_save(variant_name):
	print(f'Training on {variant_name}')

	# Create default model
	model = XLMRobertaRegressor(PRETRAINED_MODEL_NAME, drop_rate=0.2).to(DEVICE)

	# Define optimizer
	optimizer = AdamW(model.parameters(),
					  lr=1e-3,
					  eps=1e-8)

	# Define scheduler
	epochs = MAX_EPOCHS
	total_steps = len(dataloaders[TWEET][TRAIN]) * epochs
	scheduler = get_linear_schedule_with_warmup(optimizer,
												num_warmup_steps=0, num_training_steps=total_steps)

	# Define loss function
	loss_function = nn.MSELoss()

	# Fine-tune on one of our datasets
	model = train(model, optimizer, scheduler, loss_function, epochs,
				  dataloaders[variant_name][TRAIN], DEVICE, clip_value=2)

	# Save the model's learned state so it can be recovered for evaluation
	torch.save(model.state_dict(), SAVED_MODEL_PATH + variant_name)

	# Delete the model to free up GPU memory space
	del model


"""Train, save each version of the fine tuned model."""

for name in [TWEET]:
	train_and_save(name)

"""## Evaluation

### Reload models

The three models can also be downloaded from these VIEW-ONLY links. These models are around 1GB in size. 

- `model_state_dict_combined`: https://drive.google.com/file/d/1-E66ega1p2xWBU2sVAjMU4I1jZX17cqy/view?usp=sharing 
- `model_state_dict_reddit`: https://drive.google.com/file/d/1-2taTlLRWZrcDA6Z2jgtOUzd_U74YHFF/view?usp=sharing 
- `model_state_dict_twitter`: https://drive.google.com/file/d/1HGkVfRGPVORGO1qCAEXxGVCjpjPtRKpt/view?usp=sharing
"""

model_variants = {}

for name in [TWEET, REDDIT, COMBINED]:
	model = XLMRobertaRegressor(PRETRAINED_MODEL_NAME, drop_rate=0.2).to(DEVICE)
	model.load_state_dict(torch.load(SAVED_MODEL_PATH + name, map_location=torch.device('cpu')))
	model.eval()

	model_variants[name] = model

model_variants.keys()

"""### Evaluation Loop"""


def evaluate(model, dataloader, device):
	# Deactivate the model gradients so the model will save time and not learn from the test data
	with torch.no_grad():
		model.to(device)

		model.eval()

		predicted_labels = []
		true_labels = []
		for step, batch in enumerate(tqdm(dataloader)):
			batch_inputs, batch_masks, batch_labels = tuple(
				batch[field].to(device) for field in batch)

			predicted_labels += [p for p in model(batch_inputs, batch_masks)]
			true_labels += [l.data.detach().cpu().tolist() for l in batch_labels]

			del batch_inputs, batch_masks, batch_labels

		return predicted_labels, true_labels


"""Evaluate and report on accuracy score metrics"""

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from seqeval.metrics import classification_report
from tqdm import tqdm


def eval_score(train_name, test_name):
	test_preds, test_truth = evaluate(model_variants[train_name], dataloaders[test_name]['test'], device=DEVICE)
	test_preds = np.squeeze(np.asarray([p.cpu().numpy() for p in test_preds]))
	test_truth = np.array(test_truth)

	print("Trained on:", train_name)
	print("Tested on:", test_name)

	print("Pearson's R score:", pearsonr(test_preds, test_truth)[0])
	print("Spearman's rho score:", spearmanr(test_preds, test_truth)[0])
	print("MSE:", ((test_preds - test_truth) ** 2).mean())
	print("RMSE:", (np.sqrt(((test_preds - test_truth) ** 2).mean())).item())
	print("MAE:", (np.abs(test_preds - test_truth)).mean().item())


"""Evaluate each dataset's test split on the same dataset's train split and the other dataset's train split."""

eval_score(TWEET, TWEET)
# eval_score(TWEET, REDDIT)
# eval_score(TWEET, COMBINED)
# eval_score(REDDIT, TWEET)
# eval_score(REDDIT, REDDIT)
# eval_score(REDDIT, COMBINED)
# eval_score(COMBINED, TWEET)
# eval_score(COMBINED, REDDIT)
# eval_score(COMBINED, COMBINED)

"""## Base Line

Baseline models to test:
1. BERT (Devlin et al., 2018): multilingual
BERT model.
2. XLM-R (Conneau et al., 2019): multilingual
RoBERTa model.
3. XLM-T (Barbieri et al., 2021): Multilingual
RoBERTa model trained over 200M tweets.
4. DistillBERT (Sanh et al., 2019): Multilingual
distilled BERT model.
5. MiniLM (Wang et al., 2020): Multilingual
MiniLM model.

(Liu et al., 2019) found that XLM-T achieved the best performance over 7 languages, suggesting that domain specific language model training is beneficial for our tweet intimacy analysis task
(Liu et al., 2019) found that for zero-shot tasks, models has varying performances for different languages: The zero-shot performance is generally lower compared with the tasks with in-domain training. This suggests that the zero-shot task is challenging. We should explore different strategies to improve the zero-shot intimacy prediction performance.

The baseline testing was performed in a separate notebook, available at
https://github.com/aishafarooque/Tweet-Intimacy-Analysis/blob/main/Baseline_BERT_Regression.ipynb

## State of the Art
State of the art model should be regarded as XLM-T mode : Multilingual RoBERTa model trained over 200M tweets. Although this model in particular perfromed poorly on languages; Chinese, Hindi, Dutch, Korean,
"""