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

#    for param in model.parameters():
#        param.requires_grad = False

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.classifier = torch.nn.Linear(in_features=768, out_features=10, bias=True)
#	for name, param in model.named_parameters():
#		print(name, param.size(), param.requires_grad)

    model.to(device)

    batch_size = 16
    train_dataset = CIFAR10("../datasets/", train=True, transform=transform, download=True)
#    train_dataset = torch.utils.data.Subset(train_dataset, indices=range(400))
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

    validation_dataset = CIFAR10("../datasets/", train=False, transform=transform)
    validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

    print(f"train data size: {len(train_dataset)}")
    print(f"eval data size: {len(validation_dataset)}")
    print(f"batch size: {batch_size}")
    print(f"training batches: {len(train_dataloader)}")
    print(f"eval batches: {len(validation_dataloader)}")

    cifar_labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


    loss_function = torch.nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    learning_rate = 0.1
    epochs = 40
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=epochs)


    for epoch in range(epochs):
        model.train()
        print(f"training epoch {epoch + 1}")
        train_loss = 0.0
        train_correct = 0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images)
            predictions = torch.argmax(outputs.logits, 1)

            loss = loss_function(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            #print(f"p: {predictions}")
            #print(f"l: {labels}")
            correct_predictions = torch.sum(predictions == labels)
            #print(f"{correct_predictions} out of {batch_size} predictions correct in this batch")
            train_correct += correct_predictions
            if i % 500 == 499:
                print(f"batch: {i + 1}")


        train_accuracy = train_correct / len(train_dataset)

        print(f"epoch training done!")
        print(f'[{epoch + 1}, {i + 1}] train_loss: {train_loss / len(train_dataloader)}, train_accuracy: {train_accuracy}')

        model.eval()
        model.to(device)
        print(f"evaluating epoch {epoch + 1}")
        eval_loss = 0.0
        eval_correct = 0

        for i, (images, labels) in enumerate(validation_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs.logits, 1)

            correct_predictions = torch.sum(predictions == labels)
            eval_correct += correct_predictions

            loss = loss_function(outputs.logits, labels)
            eval_loss += loss.item()
	
        eval_accuracy = eval_correct / len(validation_dataset)

        print(f"epoch evaluation done!")
        print(f'[{epoch + 1}, {i + 1}] eval_loss: {eval_loss / len(validation_dataloader)}, eval_accuracy: {eval_accuracy}')


    print("done!")
    print(f'train_loss: {train_loss / len(train_dataloader)}, train_accuracy: {train_accuracy}')
    print(f'eval_loss: {eval_loss / len(validation_dataloader)}, eval_accuracy: {eval_accuracy}')


    torch.save(model.state_dict(), f"vit_epoch{epoch+1}_acc{eval_accuracy}_loss{eval_loss / len(validation_dataloader)}.pth")


if __name__ == "__main__":
    main()
