import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import sys
np.set_printoptions(threshold=sys.maxsize)

class FaceImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        self.split = split
        print(splitdir)
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        imgname = str(self.samples[index])
        img = np.array(Image.open(imgname).convert("RGB"))

        head, picname = imgname.split(self.split) # /kodak/kodim_01.png
        maskname = head +'mask'+ self.split  + picname  # /kodakmask/kodim_01.png

        # head, picname= imgname.split('Kodak')
        # maskname = head + 'kodak_mask' + picname[:-4]+'.png'
        mask = np.array(Image.open(maskname).convert('L'))
        mask = mask.reshape((img.shape[0], img.shape[1], 1)).astype(np.uint8)
        img = Image.fromarray(np.concatenate((img, mask), axis=2))
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
