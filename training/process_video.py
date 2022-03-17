# %%
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import itertools

from PIL import Image
real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

# %%
def read_json(file):
    out = dict()
    with open(file) as f:
        data = json.load(f)
    if len(data['regions']) > 1:
        out['file'] = f'{dir_path}/video/' + data['asset']['parent']['name']
        out['time'] = data['asset']['timestamp']
        
        regions = []
        for region in data['regions']:
            label = region['tags'][0]
            box = region['boundingBox']
            id = region['id']
            regions.append([label, box, id])
        out['regions'] = regions
    return out

def load_all_json():
    files = os.listdir(f'{dir_path}/output')
    annots = []
    for file in files:
        if file != 'wling.vott':
            annot = read_json(f'{dir_path}/output/{file}')
            if len(annot) > 0:
                annots.append(annot)
    return annots

# %%
def to_idx(coord):
    return int(np.floor(coord))

def coords_transform(
    old_frame_h,
    old_frame_w,
    new_frame_h,
    new_frame_w,
    i, j, h, w,
    scale: bool
    ):

    i, j, h, w = float(i), float(j), float(h), float(w)

    if scale:
        scale_h = new_frame_h/old_frame_h
        scale_w = new_frame_w/old_frame_w
        return i*scale_h, j*scale_w, h*scale_h, w*scale_w
    else:
        diff_h = (new_frame_h - old_frame_h)/2
        diff_w = (new_frame_w - old_frame_w)/2
        return i+diff_h, j+diff_w, h, w

# %%
def img_transform(img, new_h, new_w, scale: bool):
    if img.shape[0] == new_h and img.shape[1] == new_w:
        return img
    else:
        if scale:
            img_resize = cv2.resize(img, (new_w, new_h))
        else:
            add_h = to_idx((new_h - img.shape[0])/2) 
            add_w = to_idx((new_w - img.shape[1])/2)

            if add_h <= 0 and add_w <= 0:
                end_w = None if add_w == 0 else add_w
                end_h = None if add_h == 0 else add_h
                img_resize = img[-add_h:end_h, -add_w:end_w, :]
            elif add_h >= 0 and add_w >= 0:
                end_w = None if add_w == 0 else -add_w
                end_h = None if add_h == 0 else -add_h
                img_resize = np.zeros((new_h, new_w, img.shape[2]), np.dtype('uint8'))
                img_resize[add_h:end_h, add_w:end_w, :] = img
            else:
                img_resize = img
                img_resize = img_transform(img_resize, img_resize.shape[0], new_w, scale=False)
                img_resize = img_transform(img_resize, new_h, new_w, scale=False)
    
        return img_resize

# %%
def vid_to_train_data(annotations: list, out_res = 720, color = True):
    new_annotations = []
    output = []
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

        old_h = frame.shape[0]
        old_w = frame.shape[1]
        max_width = old_h if old_h < old_w else old_w
        
        if color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            frame = frame.reshape((old_h, old_w, 1))

        if old_h == old_w:
            frame = img_transform(frame, out_res, out_res, scale=True)
        elif old_h == out_res or old_w == out_res:
            frame = img_transform(frame, out_res, out_res, scale=False)
        else:
            frame = img_transform(frame, max_width, max_width, scale = False)
            frame = img_transform(frame, out_res, out_res, scale = True)
        
        if not color:
            frame = frame.reshape((out_res, out_res))

        for label, region, id in annot['regions']:
            i = region['top']
            j = region['left']
            h = region['height']
            w = region['width']

            if old_h == old_w:
                i, j, h, w = coords_transform(old_h, old_w, out_res, out_res, i, j, h, w, scale=True)
            elif old_h == out_res or old_w == out_res:
                i, j, h, w = coords_transform(old_h, old_w, out_res, out_res, i, j, h, w, scale=False)
            else:
                i, j, h, w = coords_transform(old_h, old_w, max_width, max_width, i, j, h, w, scale=False)
                i, j, h, w = coords_transform(max_width, max_width, out_res, out_res, i, j, h, w, scale=True)

            i = to_idx(i) - 1
            j = to_idx(j) - 1
            h = to_idx(h)
            w = to_idx(w)

            # frame[i,:] = 255
            # frame[i+h,:] = 255
            # frame[:,j] = 255
            # frame[:,j+w] = 255

            new_annotations.append({'id':id, 'label':label, 'row':i, 'column':j, 'height':h, 'width':w})
            output.append([id, frame])

    return output, new_annotations

# %%
def create_dataset(color, res):
    colorStr = 'RGB' if color else 'GRAY'
    print(f'creating {colorStr} images with resolution {res}x{res} ...')
    all_files = load_all_json()
    images, new_annotations = vid_to_train_data(all_files, res, color)
    try:
        os.mkdir(f'{dir_path}/images{colorStr}{res}')
        print(f'{dir_path}/images{colorStr}{res}/ created')
    except FileExistsError:
        #print(f'{dir_path}/images{colorStr}{res}/ already exists')
        pass

    with open(f'{dir_path}/annotations{colorStr}{res}.json', 'w') as fp:
        json.dump(new_annotations, fp,  indent=4)
    for id, array in images:
        im = Image.fromarray(array)

        im.save(f'{dir_path}/images{colorStr}{res}/{id}.jpg')
        
# %%
if __name__ == "__main__":
    color = (True,)
    res = (720,)
    [create_dataset(*config) for config in list(itertools.product(color, res))]