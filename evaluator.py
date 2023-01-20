import torch
from transformers import AutoTokenizer

from model import XLMRobertaRegressor

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from seqeval.metrics import classification_report
from tqdm import tqdm


def get_eval_model(fold, DEVICE, SAVED_MODEL_PATH="saved_models/"):
	PRETRAINED_MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
	MODEL_NAME = PRETRAINED_MODEL_NAME.split("/")[-1]
	model = XLMRobertaRegressor(PRETRAINED_MODEL_NAME, drop_rate=0.2).to(DEVICE)
	model.load_state_dict(torch.load(SAVED_MODEL_PATH + MODEL_NAME + "_" + str(fold), map_location=torch.device('cpu')))
	model.eval()
	return model


def evaluate(model, dataloader, device):
	PRETRAINED_MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
	tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
	# Deactivate the model gradients so the model will save time and not learn from the test data
	with torch.no_grad():
		model.to(device)

		model.eval()

		predicted_labels = []
		true_labels = []
		input_ids_docs = []
		for step, batch in enumerate(tqdm(dataloader)):
			batch_inputs, batch_masks, batch_labels = tuple(
				batch[field].to(device) for field in batch)

			predicted_labels += [p for p in model(batch_inputs, batch_masks)]
			true_labels += [l.data.detach().cpu().tolist() for l in batch_labels]

			input_ids_docs += tokenizer.batch_decode(batch_inputs)


			del batch_inputs, batch_masks

		return predicted_labels, true_labels, input_ids_docs


def eval_score(fold, dataloader, split, device=None):
	if device is None:
		DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	else:
		DEVICE = device
	if split == "test":
		res = []
		for i in range(fold):
			model = get_eval_model(i, device)
			current_test_preds, current_test_truth, docs = evaluate(model, dataloader[split], device=DEVICE)
			current_test_preds = np.squeeze(np.asarray([p.cpu().numpy() for p in current_test_preds]))
			current_test_truth = np.array(current_test_truth)
			res.append(current_test_preds)
			test_truth = current_test_truth
		test_preds = np.array(res).mean(axis=0)
	else:
		model = get_eval_model(fold, device)
		test_preds, test_truth, docs = evaluate(model, dataloader[split], device=DEVICE)
		test_preds = np.squeeze(np.asarray([p.cpu().numpy() for p in test_preds]))
		test_truth = np.array(test_truth)
	score_dict = {}
	for doc, pred, truth in zip(docs, test_preds, test_truth):
		cleaned_doc = doc.replace("<s>", "").replace("<pad>", "")
		cleaned_doc = " ".join([word.replace(" ", "") for word in cleaned_doc.split("<unk>")])
		score_dict[cleaned_doc] = [abs(pred - truth), truth, pred]
	sorted_score_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1][0], reverse=True)}
	if split == "test":
		filename = "res_test.txt"
	else:
		filename = "res_" + str(fold) + ".txt"

	with open(filename, 'w+') as f:
		f.write('Document\tDiff\tTruth\tPred\n')
		for key, value in sorted_score_dict.items():
			res_val = []
			for val in value:
				res_val.append(str(val))
			f.write('%s\t%s\n' % (key, "\t".join(res_val)))

	print("Pearson's R score:", pearsonr(test_preds, test_truth)[0])
	print("Spearman's rho score:", spearmanr(test_preds, test_truth)[0])
	print("MSE:", ((test_preds - test_truth) ** 2).mean())
	print("RMSE:" ,(np.sqrt ((  (test_preds - test_truth) ** 2).mean())).item())
	print("MAE:" ,(np.abs(test_preds - test_truth)).mean().item())

	return pearsonr(test_preds, test_truth)[0]