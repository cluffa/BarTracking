# %%
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PlateDataset:

    def __init__(self, res = 320, train = True):
        self.res = res
        self.train = train
        self.images = os.listdir('./train_data/images')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: slice):
        img = Image.open(f'./train_data/images/{self.images[idx]}')
        mask = Image.open(f'./train_data/masks/mask_{self.images[idx]}')

        if self.train:
            img = np.array(img)
            mask = np.array(mask)#[:, :, :2]
            train_transform = A.Compose(
                [
                    A.geometric.transforms.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=45, p=0.25),
                    A.RandomResizedCrop(self.res, self.res, scale=(0.2, 1.5), ratio=(0.8, 1.2)),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ]
            )

            transformed = train_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            img = img.resize((self.res, self.res))
            mask = mask.resize((self.res, self.res))
            img = np.array(img)
            mask = np.array(mask)#[:, :, :2]

        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        return img, mask
# %%
