import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10


def plot_image_selection(dataset, count):
    fig, ax = plt.subplots(1, count, figsize=(8, 3))
    classes_seen = set()
    image_idx = random.randint(0, len(dataset))
    image_class = dataset[image_idx][1]
    for i in range(count):
        while image_class in classes_seen:
            image_idx = random.randint(0, len(dataset))
            image_class = dataset[image_idx][1]
        classes_seen.add(image_class)
        ax[i].imshow(np.reshape(dataset[image_idx][0], (32, 32, 3)))
        ax[i].axis('off')

    plt.savefig('./cifar10-images.pdf', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
    dataset = CIFAR10("datasets/", train=False, download=True)
    plot_image_selection(dataset, 10)