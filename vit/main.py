from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from PIL import Image
import requests

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():

	transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])

	dataset = CIFAR10("../datasets/", train=False, transform=transform)
	dataloader = DataLoader(dataset, shuffle=False, batch_size=100, num_workers=2)

	url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
	image = Image.open(requests.get(url, stream=True).raw)

	feature_extractor = ViTFeatureExtractor(size=32)
	feature_extractor = feature_extractor.from_pretrained('google/vit-base-patch16-224')
	
	config = ViTConfig(image_size=32)
	model = ViTForImageClassification(config=config)
	model = model.from_pretrained('google/vit-base-patch16-224')
	
	cifar_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

	correct = 0
	total = 0

	for images, labels in dataloader:
		# print(cifar_labels[labels[0]])
		image_list = [img for img in images]

		inputs = feature_extractor(images=image_list, return_tensors="pt")
		outputs = model(**inputs)
		logits = outputs.logits
		# model predicts one of the 1000 ImageNet classes
		print(logits.shape)

		predicted_class_idx = logits.argmax(-1)
		print(predicted_class_idx.shape)
		for cls, label in zip(predicted_class_idx, labels):
			for cifar_label in cifar_labels:
				if cifar_label in model.config.id2label[cls.item()]:
					if cifar_label == label:
						correct += 1
					# print("Predicted class:", cifar_label)
					total += 1
		if total >= 200:
			break

	print("percentage correct ", 100 * correct / total)


if __name__ == "__main__":
	main()