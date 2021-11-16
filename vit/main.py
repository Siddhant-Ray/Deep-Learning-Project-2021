from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from PIL import Image

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

import torch.optim as optim

def main():
	print(f"GPU: {torch.cuda.is_available()}")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)

	transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])
	model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

	for param in model.parameters():
		param.requires_grad = False

	model.classifier = torch.nn.Linear(in_features=768, out_features=10, bias=True)
#	for name, param in model.named_parameters():
#		print(name, param.size(), param.requires_grad)

	model.to(device)

	train_dataset = CIFAR10("../datasets/", train=True, transform=transform, download=True)
	train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=100, num_workers=2)

	dataset = CIFAR10("../datasets/", train=False, transform=transform)
	dataloader = DataLoader(dataset, shuffle=False, batch_size=100, num_workers=2)

	cifar_labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	model.train()
	for epoch in range(1):
		print(f"Epoch: {epoch}")
		running_loss = 0.0

		for i, (images, labels) in enumerate(train_dataloader):
			print(model.device)

			images = images.to(device)
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			image_list = [img for img in images]

			print("forward")
			outputs = model(images)

			print("backward")
			loss = loss_function(outputs.logits, labels)
			loss.backward()
			print("step")
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 10 == 9:
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0

			logits = outputs.logits

			predicted_class_idx = logits.argmax(-1)


if __name__ == "__main__":
	main()