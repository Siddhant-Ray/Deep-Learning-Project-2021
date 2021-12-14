from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np
import os

random.seed(0)

transform = transforms.Compose([
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
])

source_sets = {
    "train": CIFAR10("../datasets/", train=True, transform=transform, download=True),
    "eval": CIFAR10("../datasets/", train=False, transform=transform, download=True)
}

# cifar10_loader = DataLoader(cifar10_set, shuffle=False, batch_size=len(), num_workers=2)

for mode in ("train", "eval"):
    dset = source_sets[mode]
    N = len(dset)
    OUT_DIR = "../datasets/combined_cifar_" + mode + "/"
    OUT_SIZE = 100

    os.mkdir(OUT_DIR)

    all_logits = np.zeros((OUT_SIZE, 10))
    for i in range(OUT_SIZE):
        # selected_images_idx = random.sample(range(N), 4)
        # random.shuffle(selected_images_idx)
        selected_images = []
        selected_labels = []

        while len(selected_images) < 4:
            rand_img, l = dset[random.randint(0, N - 1)]
            if l in selected_labels:
                continue
            selected_labels.append(l)
            selected_images.append(rand_img)

        # selected_images = [cifar10_set[j][0] for j in selected_images_idx]
        # selected_labels = [cifar10_set[j][1] for j in selected_images_idx]
        combined_image = Image.new("RGB", (64, 64))
        combined_image.paste(selected_images[0], (0, 0))
        combined_image.paste(selected_images[1], (32, 0))
        combined_image.paste(selected_images[2], (0, 32))
        combined_image.paste(selected_images[3], (32, 32))

        combined_image.save(OUT_DIR + str(i).zfill(5) + ".jpg")
        for l in selected_labels:
            all_logits[i][l] = 1

    np.savetxt(OUT_DIR + "labels.csv", all_logits, fmt="%d")