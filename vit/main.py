from transformers import ViTForImageClassification

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

import torch.optim as optim
import numpy as np

def main():
    print(f"GPU: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Linear(in_features=768, out_features=10, bias=True)
#	for name, param in model.named_parameters():
#		print(name, param.size(), param.requires_grad)

    model.to(device)

    batch_size = 200
    train_dataset = CIFAR10("../datasets/", train=True, transform=transform, download=True)
    train_dataset = torch.utils.data.Subset(train_dataset, indices=range(400))
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

    validation_dataset = CIFAR10("../datasets/", train=False, transform=transform)
    validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

    print(f"train data size: {len(train_dataset)}")
    print(f"validation data size: {len(validation_dataset)}")
    print(f"batch size: {batch_size}")

    cifar_labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for epoch in range(5):
        print(f"Epoch: {epoch}")
        print(device)
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_function(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            if i % 10 == 9:
                print(f"batch: {i + 1}")

        print(f"Epoch: {epoch} done!")
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss / len(train_dataloader)))

    print("evaluating...")
    model.eval()
    model.to(device)

    validation_loss = 0.0
    correct = 0
    for i, (images, labels) in enumerate(validation_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        for oup, lbl in zip(outputs.logits, labels):
            if oup.argmax() == lbl.argmax():
                correct += 1
        loss = loss_function(outputs.logits, labels)
        validation_loss = loss.item() * images.size(0)

    train_loss = 0
    epoch = 5
    print(f"Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {validation_loss / len(validation_dataloader)}")
    accuracy = correct / len(validation_dataset)
    print(f"Validation Accuracy: {accuracy}")

    torch.save(model.state_dict(), f"vit_epoch{epoch + 1}_acc{accuracy}_loss{validation_loss / len(validation_dataloader)}.pth")



if __name__ == "__main__":
    main()