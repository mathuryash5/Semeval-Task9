import pandas as pd
import numpy as np
import torch
import dataset
import datasets
import sys
import dataloaders
import evaluator
import trainer
import transformers

# set up verbosity of libraries
import create_train_dev_test
from sklearn.model_selection import  StratifiedKFold

datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()


def kfold_stratified_sampling(X_train, y_train, X_test, y_test, epochs=1,
							  train_batch_size=16, eval_batch_size=16, num_split=4, device="cpu"):
	skf = StratifiedKFold(num_split)
	max_pearson_r = float("-inf")
	best_fold = 0
	test_datasets = {"test": {"X": X_test["document"], "y": X_test["label"]}}
	test_ia_datasets = dataset.get_datasets(test_datasets)
	test_dataloader = dataloaders.get_dataloaders(test_ia_datasets, train_batch_size, eval_batch_size)
	for fold, (train, valid) in enumerate(skf.split(X_train, y_train)):
		print(np.bincount(y_train[train]))
		print('train -  {}   |   valid -  {}'.format(np.bincount(y_train[train]), np.bincount(y_train[valid])))
		X_train_cpy = X_train.copy()
		y_train_cpy = X_train_cpy.loc[train]["label"]
		y_valid_cpy = X_train_cpy.loc[valid]["label"]
		X_train_cpy = X_train_cpy.drop(["label", "index"], axis=1)

		split_datasets = {"train": {"X": X_train_cpy.loc[train]["document"], "y": y_train_cpy},
						  "valid": {"X": X_train_cpy.loc[valid]["document"], "y": y_valid_cpy}}
		ia_datasets = dataset.get_datasets(split_datasets)
		dataloader = dataloaders.get_dataloaders(ia_datasets, train_batch_size, eval_batch_size)
		# trainer.train_and_save(dataloader, fold, epochs, device)
		pearson_r = evaluator.eval_score(fold, dataloader, "valid", device)
		if pearson_r > max_pearson_r:
			max_pearson_r = pearson_r
			best_fold = fold

	print("Best Fold: {}".format(best_fold))
	evaluator.eval_score(num_split, test_dataloader, "test")





if __name__ == "__main__":
	arguments = sys.argv
	filename = arguments[1]
	epochs = int(arguments[2])
	train_batch_size = int(arguments[3])
	eval_batch_size = int(arguments[4])
	if len(arguments) == 6:
		device = arguments[5]
	else:
		device = None
	split_datasets = create_train_dev_test.create_splits(filename)
	kfold_stratified_sampling(split_datasets["train"]["X"], split_datasets["train"]["y"],
							  split_datasets["test"]["X"], split_datasets["test"]["y"],
							  epochs=epochs,
							  train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, device = device)
