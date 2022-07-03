import numpy as np
import albumentations as A
import cv2

class PlateDataset:
    def __init__(self, npz_path = 'train/train_data.npz', multiply = 1):
        self.multiply = multiply

        npzfile = np.load(npz_path)
        self.imgs = npzfile['imgs']
        self.boxes = np.clip(npzfile['boxes'], 0.0, 1.0)

    def __len__(self):
        return len(self.imgs)*self.multiply

    def __getitem__(self, idx: slice):
        idx = int(idx/self.multiply)
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.5),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.5, rotate_limit=15, p=1, border_mode=cv2.BORDER_CONSTANT)],
            bbox_params = A.BboxParams(format="albumentations", label_fields=['class_id'])
        )
        
        transformed = train_transform(
            image=self.imgs[idx],
            bboxes=self.boxes[idx],
            class_id=[0, 1]
            )
        
        box = np.empty((2, 4), dtype=np.float32)

        idx = 0
        for id in transformed['class_id']:
            box[id] = transformed['bboxes'][idx]
            idx += 1

        return  transformed['image'], box

    def getall(self):
        imgs = np.empty((300, 300, 3, self.__len__()))
        boxes = np.empty((2, 4, self.__len__()))

        for i in range(0, len(self.imgs) - 1):
            temp = self.__getitem__(i)
            imgs[:, :, :, i] = temp[0]
            boxes[:, :, i] = temp[1]

        boxes = np.concatenate((boxes[0], boxes[1]), axis=0)
        imgs = imgs/255.0

        return imgs, boxes
            

# %%
if __name__ == '__main__':
    dataset = PlateDataset(multiply=2)
    img, box = dataset.getall()
    print(img.shape, box.shape)