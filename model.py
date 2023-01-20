from transformers import AutoConfig
import torch

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

def get_model():
	# set up the configuration for BERT model
	PRETRAINED_MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
	config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)

