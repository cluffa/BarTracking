# %%
import numpy as np
from torchvision import transforms
import albumentations as A
import cv2

class PlateDataset:

    def __init__(self, res = 120, train = True, npz_path = 'dev/train/train_data.npz', multiply = 1):
        self.res = res
        self.train = train
        self.multiply = multiply
        # self.images = os.listdir('dev/train/train_data/images')
        
        
        # self.imgs = [Image.open(f'./train_data/images/{fn}') for fn in self.images]
        # self.imgs = [np.array(img) for img in self.imgs]
        # self.masks = [Image.open(f'./train_data/masks/mask_{fn}') for fn in self.images]
        # self.masks = [np.array(mask) for mask in self.masks]
        
        # load train_data.npz
        npzfile = np.load(npz_path)
        self.imgs = npzfile['imgs']
        self.masks = npzfile['masks']
        #self.masks = self.masks[:, :, :, :2]

    def __len__(self):
        return len(self.imgs)*self.multiply

    def __getitem__(self, idx: slice):
        idx = int(idx/self.multiply)
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.5, rotate_limit=15, p=1, border_mode=cv2.BORDER_CONSTANT),
        ])
        
        transformed = train_transform(image=self.imgs[idx], mask=self.masks[idx])
        img = transformed['image']
        mask = transformed['mask']

        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        return img, mask
# %%
if __name__ == '__main__':
    dataset = PlateDataset(train=True)
    img, mask = dataset[0]
    print(img.shape, mask.shape)