import json
import os.path

from PIL import Image
from torch.utils.data import Dataset
from joblib import Parallel, delayed

class MedicalImageDataset(Dataset):
    def __init__(self, config, mode='train', transform=None):
        super(MedicalImageDataset, self).__init__()
        assert mode in ['train', 'test']
        self.path = config["dataset"]["path"]
        self.labels = []
        self.transform = transform if transform is not None else lambda x:x

        if not os.path.exists(config.json):
            raise FileNotFoundError(f"404 FILE NOT FOUND: {config.json}")

        with open(config.json) as f:
            img_name = json.load(f)

        if mode == "train":
            pass

    def parallel_load(self, img_dir, img_list, img_size, verbose=0):
        return Parallel(n_jobs=-1, verbose=verbose)(delayed(
            lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
                (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
