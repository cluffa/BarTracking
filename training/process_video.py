# %%
from turtle import shape
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# %%
def read_json(file):
    out = dict()
    f = open(file)
    data = json.load(f)
    if len(data['regions']) > 1:
        out['file'] = './video/' + data['asset']['parent']['name']
        out['time'] = data['asset']['timestamp']
        regions = []
        for region in data['regions']:
            label = region['tags'][0]
            box = region['boundingBox']
            regions.append([label, box])
        out['regions'] = regions
    return out

def load_all_json():
    files = os.listdir('./output')
    annots = []
    for file in files:
        if file != 'wling.vott':
            annot = read_json('./output/' + file)
            if len(annot) > 0:
                annots.append(annot)
    return annots

# %%
def to_idx(cord):
    return int(round(cord, 0))

def cords_transform(
    old_frame_w,
    old_frame_h,
    new_frame_w,
    new_frame_h,
    i, j, h, w,
    scale: bool
    ):

    if scale:
        scale_h = new_frame_h/old_frame_h
        scale_w = new_frame_w/old_frame_w
        return i*scale_h, j*scale_w, h*scale_h, w*scale_w
    else:
        diff_h = (new_frame_h - old_frame_h)/2
        diff_w = (new_frame_w - old_frame_w)/2
        return i+diff_h, j+diff_w, h, w

def img_transform(img, new_w, new_h, scale: bool):
    if img.shape[0] == new_h and img.shape[1] == new_w:
        return img
    else:
        if scale:
            img_resize = cv2.resize(img, (new_w, new_h))
        else:
            add_h = to_idx((new_h - img.shape[0])/2) 
            add_w = to_idx((new_w - img.shape[1])/2)

            if add_h <= 0 and add_w <= 0:
                img_resize = img[-add_h:add_h, -add_w:add_w, :]
            elif add_h >= 0 and add_w >= 0:
                end_w = None if add_w == 0 else -add_w
                end_h = None if add_h == 0 else -add_h
                img_resize = np.zeros((new_h, new_w, img.shape[2]), np.dtype('uint8'))
                img_resize[add_h:end_h, add_w:end_w, :] = img
            else:
                img_resize = img
                img_resize1 = img_transform(img_resize, img_resize.shape[1], new_h, scale=False)[0]
                img_resize2 = img_transform(img_resize1, new_w, img_resize1.shape[0], scale=False)[0]
        
        return img_resize, img_resize1.shape, img_resize2.shape



def vid_to_train_data(annotations: list):
    frames = []
    for annot in annotations:
        cap = cv2.VideoCapture(annot['file'])
        
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        time = annot['time']

        frame_idx = int(round(fps*time, 0)) - 4

        buf = np.empty((frame_idx + 1, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc <= frame_idx and ret):
            ret, buf[fc] = cap.read()
            fc += 1
        cap.release()

        frame = buf[-1]
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for label, region in annot['regions']:
            i = to_idx(region['top']) - 1
            j = to_idx(region['left']) - 1
            h = to_idx(region['height'])
            w = to_idx(region['width'])

            frame[i,:] = 255
            frame[i+h,:] = 255
            frame[:,j] = 255
            frame[:,j+w] = 255
        
        frames.append(frame)

    return frames

# %%
all_files = load_all_json()
imgs = vid_to_train_data(all_files[0:10])
for img in imgs:
    plt.imshow(img)
    plt.show()


# %%
