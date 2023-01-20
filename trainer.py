import torch
from torch import nn

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm
from tqdm import tqdm

from model import XLMRobertaRegressor


def train(model, optimizer, scheduler, loss_function, epochs, train_dataloader, device, clip_value=2):
	for epoch in range(epochs):
		print(f'Epoch #{epoch}')

		best_loss = 1e10
		model.train()

		for step, batch in enumerate(tqdm(train_dataloader)):
			batch_inputs, batch_masks, batch_labels = \
				tuple(batch[field].to(device) for field in batch if field != "")

			model.zero_grad()
			outputs = model(batch_inputs, batch_masks)

			outputs = outputs.to(torch.float64)
			batch_masks = batch_masks.to(torch.float64)

			loss = loss_function(outputs.squeeze(), batch_labels.squeeze())

			loss.backward()
			clip_grad_norm(model.parameters(), clip_value)
			optimizer.step()
			scheduler.step()

	return model


def train_and_save(dataloaders, fold, epochs=1, device=None, SAVED_MODEL_PATH="saved_models/"):

	# Create default model
	if device is None:
		DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	else:
		DEVICE = device
	print(DEVICE)
	PRETRAINED_MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
	MODEL_NAME = PRETRAINED_MODEL_NAME.split("/")[-1]
	model = XLMRobertaRegressor(PRETRAINED_MODEL_NAME, drop_rate=0.2).to(DEVICE)
	TRAIN = "train"
	VALID = "valid"

	# Define optimizer
	optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8)

	# Define scheduler
	total_steps = len(dataloaders[TRAIN]) * epochs
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

	# Define loss function
	loss_function = nn.MSELoss()

	# Fine-tune on one of our datasets
	model = train(model, optimizer, scheduler, loss_function, epochs, dataloaders[TRAIN], DEVICE, clip_value=2)

	# Save the model's learned state so it can be recovered for evaluation
	torch.save(model.state_dict(), SAVED_MODEL_PATH + MODEL_NAME + "_" + str(fold))

	# Delete the model to free up GPU memory space
	del model
