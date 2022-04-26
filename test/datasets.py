# %%
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import albumentations as A

class PlateDataset:

    def __init__(self, res = 320, train = True):
        self.res = res
        self.train = train
        self.images = os.listdir('./train_data/images')
        
        
        self.imgs = [Image.open(f'./train_data/images/{fn}') for fn in self.images]
        self.imgs = [np.array(img) for img in self.imgs]
        self.masks = [Image.open(f'./train_data/masks/mask_{fn}') for fn in self.images]
        self.masks = [np.array(mask) for mask in self.masks]

    def __len__(self):
        return len(self.images)*4

    def __getitem__(self, idx: slice):
        idx = int(idx/4)
        
        #img = Image.open(f'./train_data/images/{self.images[idx]}')
        #mask = Image.open(f'./train_data/masks/mask_{self.images[idx]}')

        if self.train:
            #img = np.array(img)
            #mask = np.array(mask)
            train_transform = A.Compose(
                [
                    A.RandomResizedCrop(self.res, self.res, scale=(0.8, 2), ratio=(0.8, 1.2)),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ]
            )

            transformed = train_transform(image=self.imgs[idx], mask=self.masks[idx])
            img = transformed['image']
            mask = transformed['mask']
        else:
            img = img.resize((self.res, self.res))
            mask = mask.resize((self.res, self.res))
            img = np.array(img)
            mask = np.array(mask)

        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        return img, mask
# %%
if __name__ == '__main__':
    dataset = PlateDataset(train=True)
    img, mask = dataset[0]
    print(img.shape, mask.shape)