from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from random import randrange
from PIL import Image
import numpy as np
import os

random.seed(0)

transform = transforms.Compose([
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
])

path = "../DemystifyLocalViT/CIFAR10_data/cifar-10-batches-py/"

source_sets = {
    "train": CIFAR10(path, train=True, transform=transform, download=True),
    "eval": CIFAR10(path, train=False, transform=transform, download=True)
}

background_paths = ["background/" + x for x in os.listdir('background/')]
number_of_images = len(background_paths)


for mode in ("train", "eval"):
    dset = source_sets[mode]
    N = len(dset)
    OUT_DIR = "../DemystifyLocalViT/CIFAR10_data/background_cifar_" + mode + "/"

    # Go over each background image

    OUT_SIZE = number_of_images * 10 # number of backgrounds * number of labels
    os.makedirs(OUT_DIR, exist_ok=True)
    all_logits = np.zeros((OUT_SIZE, 10))

    i = 0
    for path in background_paths:
        # for each class
        for x in range(10):

            background = Image.open(path)
            width, height = background.size

            assert width >= 128
            assert height >= 128

            size = 128
            x1 = randrange(0, width - size)
            y1 = randrange(0, height - size)

            # randomly cropped image of size 128
            background = background.crop((x1, y1, x1 + size, y1 + size))


            class_img = []
            l = 0
            while True:
                class_img, l = dset[random.randint(0, N - 1)]
                if l == x:
                    break
            img = background
            x_pos = randrange(0,128-32)
            y_pos = randrange(0,128-32)

            img.paste(class_img, (x_pos, y_pos))
            img.save(OUT_DIR + str(i).zfill(5) + ".jpg")
            all_logits[i][l] = 1
            i += 1



    np.savetxt(OUT_DIR + "labels.csv", all_logits, fmt="%d")
