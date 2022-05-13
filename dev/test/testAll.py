import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

plt.style.use('seaborn-whitegrid')

res = 320
device = 'cpu'

def create_gif(fp_in, savepath, frameReduction=1):
    model = torch.load(f'dev/train/best_model.pth', map_location=device)

    vidcap = cv2.VideoCapture(fp_in)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameRate = int(vidcap.get(cv2.CAP_PROP_FPS))
    video = torch.empty((frameCount, 3, res, res))

    if frameCount == 0:
        return 'Failed, No frames found'
        
    success = True
    i = 0
    while success:
        success, frame = vidcap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (res, res))
            frame = transforms.ToTensor()(frame)
            video[i] = frame
            i += 1
            
    dataloader = DataLoader(video, batch_size=64)
    mask = torch.empty((frameCount, 2, res, res), dtype=torch.float64)
    filledTo = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            size = batch.shape[0]
            pred = model(batch)

            # add pred to mask
            mask[filledTo:(filledTo + size)] = pred.cpu().detach()
            filledTo += size

        #del batch, pred, model
        torch.cuda.empty_cache()

        mask = mask.numpy()
        video = video.numpy()
        
    centers = []
    step = frameReduction
    imgs = []
    for idx in range(0, frameCount, step):
        size = 16
        ratio = 4
        plt.figure(figsize=(size, size/ratio))

        loc = 141

        ax1 = plt.subplot(loc)
        ax1.set_axis_off()
        ax1.imshow(video[idx].transpose(1, 2, 0))
        ax1.set_title('Video Input')

        ax2 = plt.subplot(loc + 1)
        ax2.set_axis_off()
        ax2.imshow(mask[idx, 0] - mask[idx, 1], cmap='bwr')
        ax2.set_title('Prediction Mask')
        
        ax3 = plt.subplot(loc + 2)
        ax3.set_axis_off()
        ax3.set_title('Fit Ellipse and Center')
        
        img = np.zeros((res, res))
        
        center = np.empty((2, 2))
        
        # inside ellipse
        try:
            contours, _ = cv2.findContours(np.array(mask[idx, 0]*255, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            largest = np.argmax([cv2.contourArea(c) for c in contours])
            ellipse = cv2.fitEllipse(contours[largest])
            cv2.ellipse(img, ellipse, color=(1, 0, 0), thickness=2)
            
            ax3.axhline(y=ellipse[0][1], color='r', linestyle='--', linewidth=1)
            ax3.axvline(x=ellipse[0][0], color='r', linestyle='--', linewidth=1)
            
            center[0] = ellipse[0]
        except Exception as e:
            print('error at index', idx, ':\n', e)
        
        # outside ellipse
        try:
            contours, _ = cv2.findContours(np.array(mask[idx, 1]*255, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            largest = np.argmax([cv2.contourArea(c) for c in contours])
            ellipse = cv2.fitEllipse(contours[largest])
            cv2.ellipse(img, ellipse, color=(-1, 0, 0), thickness=2)
            
            ax3.axhline(y=ellipse[0][1], color='r', linestyle='--', linewidth=1)
            ax3.axvline(x=ellipse[0][0], color='r', linestyle='--', linewidth=1)
            
            center[1] = ellipse[0]
        except Exception as e:
            print('error at index', idx, ':\n', e)
            
        ax3.imshow(img, cmap='bwr')
        
        ax4 = plt.subplot(loc + 3)
        ax4.set_axis_off()
        
        centers.append(center)
        
        data = np.array(centers)
        ax4.imshow(np.zeros((res, res)))
        ax4.plot(data[:, 0, 0], data[:, 0, 1], 'r', linewidth=1)
        ax4.plot(data[:, 1, 0], data[:, 1, 1], 'b', linewidth=1)
        ax4.plot(data[:,:,0].mean(axis=1),data[:,:,1].mean(axis=1), 'g', linewidth=2)
        
        
        ax4.set_xbound(0, res)
        ax4.set_ybound(0, res)
        ax4.set_title('Track Path and Mean')
        

        plt.savefig(f'dev/test/tmp/{idx}.png', bbox_inches='tight', facecolor = 'white', pad_inches = 0.2)
        plt.close()
        imgs.append(Image.open(f'dev/test/tmp/{idx}.png'))

    imgs[0].save(savepath, format='GIF', append_images=imgs[1:], save_all=True, duration=step*1000/frameRate, loop=0)
    return 'Done'

if __name__ == '__main__':
    videos = os.listdir('dev/test/videos')
    for idx, video in enumerate(videos):
        success = create_gif(f'dev/test/videos/{video}', f'dev/test/test_gifs/{video.split(".")[0]}.gif', frameReduction=1)
        print(f'[{idx+1}/{len(videos)}] {video} {success}')