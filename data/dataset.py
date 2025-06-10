import json
import os.path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
import argparse
import yaml

class MedicalImageDataset(Dataset):
    def __init__(self, config, mode='train', transform=None):
        super(MedicalImageDataset, self).__init__()
        assert mode in ['train', 'test']
        self.path = config["dataset"]["path"]
        self.json = config["dataset"]["json"]
        self.image_size = config["dataset"]["image_size"]
        self.images = []
        self.labels = []
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if not os.path.exists(self.json):
            raise FileNotFoundError(f"404 FILE NOT FOUND: {self.json}")

        with open(self.json) as f:
            self.img_name = json.load(f)

        if mode == "train":
            normal = self.img_name["train"]["0"]
            self.images += normal
            self.labels += len(normal) * [0]

        elif mode == "test":
            normal = self.img_name["test"]["0"]
            anomalies = self.img_name["test"]["1"]
            data = normal + anomalies

            self.images += data
            self.labels += len(normal) * [0] + len(anomalies) * [1]

    def __getitem__(self, item):
        img_path = os.path.join(self.path, self.images[item])
        img = Image.open(img_path).convert("L").resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        img = self.transform(img)
        label = self.labels[item]
        return img, label

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='VinCXR', help='Dataset name')

    args = parser.parse_args()
    config_path = os.path.join('../config', f"{args.config}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"404 FILE NOT FOUND: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["dataset"]["json"] = "VinCXR/data.json"
    config["dataset"]["path"] = "VinCXR/images"
    transforms = transforms.Compose([])

    dataset = MedicalImageDataset(config, mode="train", transform=transforms, abnormal_ratio=0.2)
    print(f"Sum images in dataset: {len(dataset)}")
    image, label = dataset[12]
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()