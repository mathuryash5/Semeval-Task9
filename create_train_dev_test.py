import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

import dataloaders
import dataset
import trainer


def load_data(filename):
	twitter_df_train = pd.read_csv(filename, on_bad_lines='skip')
	twitter_df_train = twitter_df_train.rename(columns={'text': 'document', 'label': 'label'})

	print("Dataset size:", len(twitter_df_train), '\n')
	# print(twitter_df_train.info())
	return twitter_df_train


def split_dataset(dataset, train_size, test_size):
	dataset["language_code"] = dataset.language.astype('category').cat.codes
	X = dataset[['document', 'label']]
	y = dataset['language_code']

	# Bin size = 1.0
	bins = np.linspace(start=1.0, stop=5.0, num=5)

	language_codes = dataset["language_code"].tolist()
	# language_codes.sort()

	# binned_y = np.digitize(y, language_codes)
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=language_codes, train_size=train_size,
														random_state=42)

	print(y_train.value_counts())
	print(y_test.value_counts())

	X_train.reset_index(inplace=True)
	y_train = y_train.reset_index()
	y_train = y_train.language_code
	# y_train = X_train["label"]
	# X_train = X_train.drop(["label"], axis=1)
	# y_test = X_test["label"]
	# X_test = X_test.drop(["label"], axis=1)

	return {'train': {'X': X_train, 'y': y_train},
			'test': {'X': X_test, 'y': y_test}}


# def kfold_stratified_sampling(X_train, y_train, X_test=None, y_test=None, train_batch_size = 16, eval_batch_size = 16, num_split=4):
# 	skf = StratifiedKFold(num_split)
# 	for fold, (train, valid) in enumerate(skf.split(X_train, y_train)):
# 		print(np.bincount(y_train[train]))
# 		print('train -  {}   |   valid -  {}'.format(np.bincount(y_train[train]), np.bincount(y_train[valid])))
# 		y_train = X_train.loc[train]["label"]
# 		y_valid = X_train.loc[valid]["label"]
# 		X_train = X_train.drop(["label", "index"], axis=1)
#
# 		split_datasets = {"train": {"X": X_train.loc[train]["document"], "y": y_train},
# 						  "valid": {"X": X_train.loc[valid]["document"], "y": y_valid}}
# 		ia_datasets = dataset.get_datasets(split_datasets)
# 		dataloader = dataloaders.get_dataloaders(ia_datasets, train_batch_size, eval_batch_size)
# 		trainer.train_and_save(dataloader, fold)
# 		break



def create_splits(filename = "train.csv"):
	df = load_data(filename)
	split_datasets = split_dataset(df, 0.8, 0.2)
	# kfold_stratified_sampling(split_datasets["train"]["X"], split_datasets["train"]["y"])
	return split_datasets