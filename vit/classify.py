from transformers import ViTForImageClassification, ViTConfig
import sys
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import datetime
import json
import torch.backends.cudnn as cudnn
from torch.nn import functional

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from adv_dataset.combined_cifar import CombinedCifar

BATCH_SIZE = 64
MODEL_FILE = "model_vit_scratch_e80_acc0.75_loss0.78.pth"
DATASET_DIR = "../datasets/combined_cifar_eval/"

def get_model(device):
    vit_conf = ViTConfig()
    vit_conf.num_labels = 10
    model = ViTForImageClassification(config = vit_conf)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.classifier = torch.nn.Linear(in_features=768, out_features=10, bias=True)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    model.to(device)
    return model

def main():
    # today = datetime.date.today()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    OUT_DIR = "classification_" + now + "/"
    os.mkdir(OUT_DIR)

    params = {
        "DATE": now,
        "BATCH_SIZE": BATCH_SIZE,
        "MODEL_FILE": MODEL_FILE,
        "DATASET_DIR": DATASET_DIR
    }

    print(f"GPU: {torch.cuda.is_available()}")
    device_id = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ConvertImageDtype(torch.float),
        #transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = CombinedCifar(DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = get_model(device)

    all_logits = np.zeros((len(dataset), 10))
    all_softmax = np.zeros((len(dataset), 10))

    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        logits = np.array(outputs.logits.cpu().detach())
        softmax = functional.softmax(outputs.logits.cpu().detach().float(), dim = 1)

        all_logits[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE] = logits
        all_softmax[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE] = softmax

    np.savetxt(OUT_DIR + "logits.csv", all_logits)
    np.savetxt(OUT_DIR + "logits_int.csv", all_logits, fmt="%d")
    np.savetxt(OUT_DIR + "softmax_probs.csv", all_softmax)
    with open(OUT_DIR + "params.json", "w") as f:
        json.dump(params, f)

if __name__ == "__main__":
    main()