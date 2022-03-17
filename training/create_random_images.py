# %%
import cv2
import numpy as np
import os
import time
from torchvision import transforms
from PIL import Image

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# %%
def select_random_frames():
    video_files = os.listdir(f'{DIR_PATH}/video2/')
    for file in video_files:
        cap = cv2.VideoCapture(f'{DIR_PATH}/video2/{file}')
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True
        while (fc < frameCount and ret):
            ret, buf[fc] = cap.read()
            fc += 1

        num_images = 10
        choices = np.random.choice(list(range(len(buf))), num_images)
        buf = buf[choices, :, :, ::-1]

        cap.release()
        for image in buf:
            image = Image.fromarray(image)
            image = transforms.Resize((720, 720))(image)
            image = transforms.RandomAffine(10)(image)
            image = transforms.RandomHorizontalFlip(0.25)(image)
            image = transforms.RandomVerticalFlip(0.25)(image)
            image = transforms.RandomGrayscale()(image)
            image.save(f'{DIR_PATH}/training_images/{time.thread_time_ns()}.jpg')

if __name__ == "__main__":
    select_random_frames()
