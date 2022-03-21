# %%
import torch
import numpy as np
import cv2

from training.initialPlateBoundingBoxModel import model as barModel
from training.initialPlateBoundingBoxModel import model as initBarModel
from training.platesClassCenterModel import model as plateModel

barModel.eval()
initBarModel.eval()
plateModel.eval()

from PIL import Image
from PIL import ImageDraw
from torchvision import transforms

def get_bbs(img, tm1pred, tm2pred, res = 256):
    tm1pred, tm2pred = None, None
    with torch.no_grad():
        init = tm1pred is None and tm2pred is None
        #img = img.resize((720, 720))
        if not init:
            tensor = transforms.Grayscale(1)(img)
            tensor = transforms.ToTensor()(tensor.resize((res,res)))

            outline1 = Image.new('RGB', (res, res), color = (0, 0, 0))
            id1 = ImageDraw.Draw(outline1)
            outline2 = Image.new('RGB', (res, res), color = (0, 0, 0))
            id2 = ImageDraw.Draw(outline2)

            if tm2pred is not None:
                id1.rectangle(tm2pred[0], fill=(128, 128, 128)) # t - 2 outside
                id2.rectangle(tm2pred[1], fill=(128, 128, 128)) # t - 2 inside

            if tm1pred is not None:
                id1.rectangle(tm1pred[0], fill=(255, 255, 255)) # t - 1 outside
                id2.rectangle(tm1pred[1], fill=(255, 255, 255)) # t - 1 inside

            outline1 = transforms.Grayscale(1)(outline1)
            outline1 = transforms.ToTensor()(outline1)

            outline2 = transforms.Grayscale(1)(outline2)
            outline2 = transforms.ToTensor()(outline2)

            tensor = torch.cat((outline1, outline2, tensor))
            tensor = tensor.reshape((1, 3, res, res))

            tensor = tensor.cuda()

            outsideBB, insideBB = barModel(tensor)
        else:
            tensor = transforms.ToTensor()(img.resize((res,res)))
            tensor = tensor.reshape((1, 3, res, res))
            tensor = tensor.cuda()
            outsideBB, insideBB = initBarModel(tensor)

        bbs = (outsideBB[0].tolist(), insideBB[0].tolist())

        outside = img.crop(bbs[0])
        inside = img.crop(bbs[1])

        outside = outside.resize((128, 128))
        inside = inside.resize((128, 128))

        outside = transforms.ToTensor()(outside)

        inside = transforms.ToTensor()(inside)

        label, center = plateModel(torch.stack((outside, inside)).cuda())
        label = label.tolist()
        center = center.tolist()

        id = ImageDraw.Draw(img) 

        
        for lab, (xC, yC), (x, y, xmax, ymax) in zip(label, center, bbs):
            w, h = xmax - x, ymax - y
            pxy = x + xC*(w/128), y + yC*(h/128)
            col = (0,255,0,0) if lab[0] < lab[1] else (255,0,0,0)

            id.ellipse(pointToBox(pxy, 5), fill = col)

        id.rectangle(bbs[0], outline=(0,255,0,0))
        id.rectangle(bbs[1], outline=(255,0,0,0))
        if tm1pred is not None:
            id.rectangle(tm1pred[0], outline=(0,255,0,0))
            id.rectangle(tm1pred[1], outline=(255,0,0,0))

        if tm2pred is not None:
            id.rectangle(tm2pred[0], outline=(0,255,0,0))
            id.rectangle(tm2pred[1], outline=(255,0,0,0))

    return img, outsideBB.cpu().numpy(), insideBB.cpu().numpy()

def pointToBox(pt, r):
    x, y = pt
    return x - r, y - r, x + r, y + r

def add_bbs_video(fp = './10000000_627962671783457_1282908874737466533_n.mp4', out = 'test.mp4'):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(out, fourcc, 30, (720,720))
    vidcap = cv2.VideoCapture(fp)

    tm2pred = None
    tm1pred = None
    success = True

    while success:
        success, image = vidcap.read()
        if success:
            image = np.array(image)
            image, outsideBB, insideBB = get_bbs(Image.fromarray(image), tm1pred, tm2pred)
            video.write(np.array(image))
            tm2pred = tm1pred
            tm1pred = outsideBB, insideBB
    video.release()

add_bbs_video('training/raw_videos_processed/199020139_3673023292803821_3870015030764879905_n.mp4', 'test1.mp4')
add_bbs_video('training/raw_videos_processed/175453678_1346199952416916_3270004720463027727_n.mp4', 'test2.mp4')
# %%
#img = Image.open('training/training_images/10374019900.jpg')
#get_bbs(img, None, None)[0]
# %%
