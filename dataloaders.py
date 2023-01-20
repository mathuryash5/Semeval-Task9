from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

def get_dataloaders(ia_datasets, train_batch_size=16, eval_batch_size=16):
	"""
	Description:
		This function implements batch training by creating dataloaders for datasets to acclerate training.
	Inputs:
		- datasets (Dict[str, Dict]): a dictionary containing a mapping between dataset names and dataset values.
		- train_batch_size (int): an integer which is used as the batch size when creating the train dataloader
		- eval_batch_size (int): an integer which is used as the batch size when creating the validation and test dataloader
	"""
	# Retrieve the dataset split labels ("train", "validation", "test")
	TRAIN_BATCH_SIZE = train_batch_size
	EVAL_BATCH_SIZE = eval_batch_size

	splits = ia_datasets.keys()

	# Choose different parameters for each dataset split
	sampler_classes = {split:
					   (RandomSampler if split == "train" else SequentialSampler)
					   for split in splits}

	batch_sizes = {split:
				   (train_batch_size if split == "train" else eval_batch_size)
				   for split in splits}

	# Initialize the BatchSamplers with the differentiated parameters
	batch_samplers = {split:
					  BatchSampler(sampler_classes[split](ia_datasets[split]),
								   batch_sizes[split],
								   drop_last=False)
					  for split in splits}

	# Build the dataloaders from the initialized BatchSamplers and the custom collate_fn
	dataloaders = {split:
				   DataLoader(ia_datasets[split],
							  batch_sampler=batch_samplers[split],
							  collate_fn=ia_datasets[split].collate_fn)
				   for split in splits}

	return dataloaders
