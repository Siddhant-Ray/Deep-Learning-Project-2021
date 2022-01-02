
import torch
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset


class CombinedCifar(Dataset):
    def __init__(self, dir, transform=None):
        self.labels = np.genfromtxt(dir + "labels.csv")
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img_path = self.dir + str(i).zfill(5) + ".jpg"
        img = read_image(img_path)
        label = torch.Tensor(self.labels[i])
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    dataset = CombinedCifar("../datasets/combined_cifar_eval/")
    print(dataset[10][0].shape, dataset[10][1])

if __name__ == "__main__":
    main()