import cv2
import numpy as np
import torch
from scipy import ndimage
import json

class segDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        super(segDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        self.data = []
        
        with open(dataset_dir, "r") as f:
          self.data = json.load(f)
        f.close()

    def __getitem__(self, idx):
        img_path = self.data[idx][0]

        image = cv2.imread(img_path)
        cls_mask = np.array(self.data[idx][1], dtype=np.uint64)

        # 90 degree rotation
        if np.random.rand() < 0.5:
          angle = np.random.randint(4) * 90
          image = ndimage.rotate(image, angle, reshape=True)
          cls_mask = ndimage.rotate(cls_mask, angle, reshape=True)

        # vertical flip
        if np.random.rand() < 0.5:
          image = np.flip(image, 0)
          cls_mask = np.flip(cls_mask, 0)
        
        # horizonal flip
        if np.random.rand() < 0.5:
          image = np.flip(image, 1)
          cls_mask = np.flip(cls_mask, 1)

        cls_mask = np.expand_dims(cls_mask, 0)

        image = cv2.resize(image, (512, 512)) / 255.0
        image = np.moveaxis(image, -1, 0)

        return torch.tensor(image).float(), torch.tensor(cls_mask.copy(), dtype=torch.float16)

    def __len__(self):
        return len(self.data)