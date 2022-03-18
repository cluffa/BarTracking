# %%
import random
import pandas as pd
import os

from PIL import Image
random.seed(121345)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(f'{DIR_PATH}/training_annotations/vott-csv-export/bar-tracking-export.csv')
new_images = 0
for idx, row in df.iterrows():
    if not os.path.isfile(f'{DIR_PATH}/plate_images/images/{row[5]}-{row[0]}'):
        fp = f'{DIR_PATH}/training_images/{row[0]}'
        im = Image.open(fp)
        crop = [int(i) for i in row[1:5]]
        rng = (-10, 50)
        rand = [-random.randint(*rng), -random.randint(*rng), random.randint(*rng), random.randint(*rng)]
        crop = [i+j for i, j in zip(crop, rand)]
        crop = [i if i <= 720 else 720 for i in crop]
        crop = [i if i >= 0 else 720 for i in crop]
        im = im.crop(crop)
        im = im.resize((128, 128))
        im.save(f'{DIR_PATH}/plate_images/images/{row[5]}-{row[0]}')
        new_images += 1

print(f'created {new_images} new plate images')