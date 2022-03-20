# %%
import torch
import numpy as np
import cv2

from training.initialPlateBoundingBoxModel import model as barModel
from training.platesClassCenterModel import model as plateModel

barModel.eval()
plateModel.eval()

from PIL import Image
from PIL import ImageDraw
from torchvision import transforms

def get_bbs(img):
    with torch.no_grad():
        #img = img.resize((720, 720))
        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        tensor = tensor.cuda()
        if len(tensor.shape) < 4:
            tensor = tensor.reshape((1, 3, 720, 720))
        outsideBB, insideBB = barModel(tensor)

        bbs = (outsideBB[0].tolist(), insideBB[0].tolist())

        outside = img.crop(bbs[0])
        inside = img.crop(bbs[1])

        outside = outside.resize((128, 128))
        inside = inside.resize((128, 128))

        outside = transforms.ToTensor()(outside)
        outside = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(outside)

        inside = transforms.ToTensor()(inside)
        inside = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inside)

        label, center = plateModel(torch.stack((outside, inside)).cuda())
        label = label.tolist()
        center = center.tolist()

        id = ImageDraw.Draw(img) 

        def pointToBox(pt, r):
            x, y = pt
            return x - r, y - r, x + r, y + r
        
        for lab, (xC, yC), (x, y, xmax, ymax) in zip(label, center, bbs):
            w, h = xmax - x, ymax - y
            pxy = x + xC*(w/128), y + yC*(h/128)
            col = (0,255,0,0) if lab[0] < lab[1] else (255,0,0,0)

            id.ellipse(pointToBox(pxy, 5), fill = col)

        id.rectangle(bbs[0], outline=(0,255,0,0))
        id.rectangle(bbs[1], outline=(255,0,0,0))
    return img
    

def add_bbs_video(fp = './10000000_627962671783457_1282908874737466533_n.mp4'):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter("./test.mp4",fourcc, 30, (720,720))
    vidcap = cv2.VideoCapture(fp)
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            image = np.array(image)
            image = get_bbs(Image.fromarray(image))
            video.write(np.array(image))
            #video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()

add_bbs_video()

# %%
