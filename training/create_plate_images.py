# %%
import random
import numpy as np
import pandas as pd
import os
import shutil

from random import randint
from multiprocessing import Pool
from PIL import Image
#random.seed(121345)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
folder = 'images'
shutil.rmtree(f'{DIR_PATH}/plate_images/{folder}/')

if not os.path.isdir(f'{DIR_PATH}/plate_images/{folder}/'):
    os.mkdir(f'{DIR_PATH}/plate_images/{folder}/')

df = pd.read_csv(f'{DIR_PATH}/training_annotations/vott-csv-export/bar-tracking-export.csv')

#new_images = 0

def main(input):
    rows = []
    for i in range(25):
        idx, row = input
        fp = f'{DIR_PATH}/training_images/{row[0]}'
        im = Image.open(fp)
        xmin, ymin, xmax, ymax = row[1:5]
        cut = True if xmin == 0 or xmax == 720 or ymin == 0 or ymax == 720 else False
        if not cut:
            xC = (xmin + xmax)/2
            yC = (ymin + ymax)/2

            h, w = ymax - ymin, xmax - xmin

            pad = 0.5
            xmin += - w*pad
            xmax += w*pad
            ymin += - h*pad
            ymax += h*pad

            im = im.crop((xmin, ymin, xmax, ymax))
            im = im.resize((256, 256))
            xC, yC = 128, 128
            
            xR = randint(-32, 32)*2
            yR = randint(-32, 32)*2

            xC, yC = xC + xR, yC + yR

            xmin, ymin, xmax, ymax = xC - 64, yC - 64, xC + 64, yC + 64

            xC, yC = xC/2, yC/2

            im = im.crop((xmin, ymin, xmax, ymax))

            im.save(f'{DIR_PATH}/plate_images/{folder}/{row[5]}-{i}{row[0]}')
            #new_images += 1
            row = {
                'image':f'{row[5]}-{i}{row[0]}',
                'xmin':xC, 'ymin':yC,
                'xmax':xC, 'ymax':yC,
                'label': 'center'
                }
            rows.append(row)
    return pd.DataFrame.from_records(rows)

if __name__ == '__main__':
    with Pool(12) as p:
        plate_info = p.map(main, df.iterrows())

    plate_info = pd.concat(plate_info)
    print(plate_info)
    plate_info.to_csv(f'{DIR_PATH}/plate_images/plate-center-export.csv', index=False)




# %%
