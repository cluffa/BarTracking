# %%
import torch
import numpy as np
import cv2

from training.plateBoundingBoxModel import inside as inBarModel, outside as outBarModel
from training.platesClassCenterModel import model as plateModel

inBarModel.eval()
outBarModel.eval()
plateModel.eval()

from PIL import Image
from PIL import ImageDraw
from torchvision import transforms

def get_bbs(img, res = 256):
    with torch.no_grad():
        tensor = transforms.ToTensor()(img.resize((res,res)))
        tensor = tensor.reshape((1, 3, res, res))
        tensor = tensor.cuda()
        
        outsideBB = outBarModel(tensor)
        insideBB = inBarModel(tensor)
        
        id = ImageDraw.Draw(img) 
        bbs = [outsideBB[0].tolist(), insideBB[0].tolist()]

        if True:
            outside = img.crop(bbs[0])
            inside = img.crop(bbs[1])

            outside = outside.resize((128, 128))
            inside = inside.resize((128, 128))

            outside = transforms.ToTensor()(outside)

            inside = transforms.ToTensor()(inside)

            label, center = plateModel(torch.stack((outside, inside)).cuda())
            label = label.tolist()
            center = center.tolist()

            for lab, (xC, yC), (x, y, xmax, ymax) in zip(label, center, bbs):
                w, h = xmax - x, ymax - y
                pxy = x + xC*(w/128), y + yC*(h/128)
                col = (0,255,0,0) if lab[0] < lab[1] else (255,0,0,0)

                id.ellipse(pointToBox(pxy, 5), fill = col)

        id.rectangle(bbs[0], outline=(0,255,0,0))
        id.rectangle(bbs[1], outline=(255,0,0,0))

    return img

def pointToBox(pt, r):
    x, y = pt
    return x - r, y - r, x + r, y + r

def add_bbs_video(fp, out):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out, fourcc, 30, (720,720))
    vidcap = cv2.VideoCapture(fp)

    success = True
    while success:
        success, image = vidcap.read()
        if success:
            image = np.array(image)
            image = Image.fromarray(image)
            image = get_bbs(image)
            video.write(np.array(image))
    video.release()

add_bbs_video('training/raw_videos_processed/199020139_3673023292803821_3870015030764879905_n.mp4', 'test1.mp4')
add_bbs_video('training/raw_videos_processed/175453678_1346199952416916_3270004720463027727_n.mp4', 'test2.mp4')

# %%
#img = Image.open('training/training_images/10374019900.jpg')
#get_bbs(img, None, None)[0]
# %%
