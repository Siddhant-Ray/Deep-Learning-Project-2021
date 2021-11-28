import numpy as np, argparse, json

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

with open('configs/config_scratch.json', 'r') as f:
    config = json.load(f)

### Hyperparamters
learning_rate = config['learning_rate']
epochs = config['num_epochs']
batch_size = config['batch_size']
weight_decay = config['weight_decay']
momentum = config['momentum']
adaptivity = config['adaptivity']
max_lr = config['max_lr']

## Transform function
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        #transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train': 
    torchvision.datasets.CIFAR10(
    root='dataset/cifar-10-batches-py', train=True, download=True, transform = transforms['train']),
    'validation': 
    torchvision.datasets.CIFAR10(
    root='dataset/cifar-10-batches-py', train=False, download=True, transform=transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0),  
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)  
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet50(pretrained=False).to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    
'''for param in model.parameters():
    param.requires_grad = False'''   

# Classifier head    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 10)).to(device)

## Model specifiers
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloaders['train']), epochs=epochs)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

def run_model(model, criterion, optimizer, num_epochs=epochs):
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            print(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_accuracy = 0 

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                accuracy = (outputs.argmax(-1) == labels).float().mean()
                total_accuracy += accuracy

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = total_accuracy / len(dataloaders[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))

        #scheduler.step()                                                
            
    return model

model_trained = run_model(model, criterion, optimizer, num_epochs=epochs)
print("======> This is the classifier with non pretrained weights")

print("Parameters used for this model")
for key, value in config.items():  
    print(key, value)

torch.save(model_trained.state_dict(), 'saved_model/pytorch/weights_scratch.h5')

# Next time 
"""
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load('saved_model/pytorch/weights.h5'))

"""







