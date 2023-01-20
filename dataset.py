import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
			'labels': torch.tensor(labels).to(torch.float64),
		}

		return input_dict


def get_ia_splits(data, pad_token_id):
	ia_splits = {split: IADataset(data[split], pad_token_id)
				 for split in data}

	return ia_splits



def get_datasets(split_datasets):
	PRETRAINED_MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

	IGNORE_INDEX = -100
	tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

	# Convert the documents into sequences of input ids in the model's vocabulary
	vocab = tokenizer.get_vocab()

	prepared_datasets = prepare_data(split_datasets, tokenizer, vocab)
	ia_datasets = get_ia_splits(prepared_datasets, pad_token_id=tokenizer.pad_token_id)

	return ia_datasets





