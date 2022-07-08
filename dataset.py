import os
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import kaggle_setup
from zipfile import ZipFile
import hashlib
from torch.utils.data import Dataset


class AnimeFace(Dataset):
    base_folder = "animefacedataset"
    img_folder = "images"
    seed = 1234
    md5 = "a0558aaf6da58b037e52e9910fa7e052"

    def __init__(self, root, download=False, split="all", transform=transforms.ToTensor()):
        self.root = root
        self.base_dir = os.path.join(self.root, self.base_folder)
        if download:
            self._download()
        self.imgs = self._get_images()
        self.total_size = len(self.imgs)
        train_inds, test_inds = self._get_split()
        self.inds = {
            "train": train_inds,
            "test": test_inds,
            "all": np.arange(self.total_size)
        }[split]
        self.transform = transform

    def _download(self):
        kaggle_setup()
        kaggle_ref = "splcher/animefacedataset"
        fpath = os.path.join(self.base_dir, kaggle_ref.split("/")[-1] + ".zip")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
                if md5 == self.md5:
                    return 0
                else:
                    raise ValueError("Zip file MD5 check fails! Please try re-downloading it.")
        else:
            os.system(f"kaggle datasets download -p {self.base_dir} {kaggle_ref}")
            print("Decompressing the downloaded file...")
            # os.system(f"unzip -q -d {download_folder} {file}")  # too slow
            with ZipFile(fpath, "r") as zf:
                zf.extractall(path=self.base_dir)

    def _get_images(self):
        return sorted([
            f for f in os.listdir(os.path.join(self.base_dir, self.img_folder))
            if f.endswith(".jpg")], key=lambda x: x.zfill(14))

    def _get_split(self):
        inds = np.arange(self.total_size)
        np.random.seed(self.seed)
        np.random.shuffle(inds)
        split_at = int(self.total_size * 0.9)
        train_inds, test_inds = inds[:split_at], inds[split_at:]
        return train_inds, test_inds

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.base_dir, self.img_folder, self.imgs[self.inds[idx]])
        with Image.open(img_path) as im:  # close the file before return
            return self.transform(im) if self.transform is not None else im


if __name__ == "__main__":
    root = os.path.expanduser("~/datasets")
    data = AnimeFace(root=root, download=True)
