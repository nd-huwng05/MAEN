import json
import os.path

from PIL import Image
from torch.utils.data import Dataset
from joblib import Parallel, delayed

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
        self.transform = transform if transform is not None else lambda x:x

        if not os.path.exists(self.json):
            raise FileNotFoundError(f"404 FILE NOT FOUND: {self.json}")

        with open(self.json) as f:
            self.img_name = json.load(f)

        if mode == "train":
            normal = self.img_name["train"]["0"]
            abnormal = self.img_name["train"]["1"]
            data = normal + abnormal

            self.images += data
            self.labels += len(normal) * [0] + len(abnormal) * [1]

        elif mode == "test":
            normal = self.img_name["test"]["0"]
            abnormal = self.img_name["test"]["1"]
            data = normal + abnormal

            self.images += data
            self.labels += len(normal) * [0] + len(abnormal) * [1]

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

    dataset = MedicalImageDataset(config, mode="test")
    print(f"Sum images in dataset: {dataset.__len__()}")
    image, label = dataset.__getitem__(0)
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()