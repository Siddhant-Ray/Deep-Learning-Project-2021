import numpy as np, argparse, json, sys, os

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.optim as optim

sys.path.insert(1, os.path.join(sys.path[0],'..'))
from adv_dataset.combined_cifar import CombinedCifar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
amp = True  # mixed precision

with open('configs/config_scratch_big.json', 'r') as f:
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
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ConvertImageDtype(torch.float),
        #transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'validation': 
    CombinedCifar("dataset/combined_cifar_eval/", transform=transforms['validation'])
}

dataloaders = { 
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)  
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = models.resnet50(*args, **kwargs)

    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=amp):
            return self.model(*args, **kwargs)
      
model = ResNet(pretrained=False, num_classes=10).to(device)
model.load_state_dict(torch.load('saved_model/pytorch/weights_scratch.h5'))

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def run_model(model, criterion):
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    all_logits = np.zeros((len(image_datasets['validation']), 10))
    all_scaled_probs = np.zeros((len(image_datasets['validation']), 10))

    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_accuracy = 0 

    for batch, (inputs, labels) in enumerate(dataloaders['validation']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(inputs)
            logit_vals = np.array(outputs.cpu().detach()) #### What we really want 
            probs = F.softmax(outputs.cpu().detach().float(), dim = 1)
            #loss = criterion(outputs, labels).item()
            all_logits[batch * batch_size : (batch + 1) * batch_size] = logit_vals
            all_scaled_probs[batch * batch_size : (batch + 1) * batch_size] = probs

        
    np.savetxt("resnet_results/logits_int.csv", all_logits, fmt="%d") #Maybe we don't do integer rounding but use softmax
    np.savetxt("resnet_results/logits.csv", all_logits)

    np.savetxt("resnet_results/softmax_probs.csv", all_scaled_probs, fmt = "%.2f")
    
    return model

combined_images_model = run_model(model, criterion)
print("======> This is the classifier on combined images")

print("======> This is the model")
print(model)

    









