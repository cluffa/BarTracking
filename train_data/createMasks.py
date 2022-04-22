# %%
from matplotlib.pyplot import draw
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# %%
# import annotations.csv
annotations = pd.read_csv('./annotations.csv')
for i in range(len(annotations)):
    row = annotations.iloc[i]
    img = Image.open(f'./images/{row["image"]}')

    # create empty mask image
    mask = Image.new('RGB', img.size, (0, 0, 0))
    maskR, maskG, maskB = mask.split()

    # draw ellipse, red is inside, green is outside
    drawR = ImageDraw.Draw(maskR)
    drawR.ellipse((row['xmin_inside_plate']*720, row['ymin_inside_plate']*720, row['xmax_inside_plate']*720, row['ymax_inside_plate']*720), fill=255)

    drawG = ImageDraw.Draw(maskG)
    drawG.ellipse((row['xmin_outside_plate']*720, row['ymin_outside_plate']*720, row['xmax_outside_plate']*720, row['ymax_outside_plate']*720), fill=255)

    drawB = ImageDraw.Draw(maskB)
    xmin = min(row['xmin_inside_plate']*720, row['xmin_outside_plate']*720)
    xmax = max(row['xmax_inside_plate']*720, row['xmax_outside_plate']*720)
    ymin = min(row['ymin_inside_plate']*720, row['ymin_outside_plate']*720)
    ymax = max(row['ymax_inside_plate']*720, row['ymax_outside_plate']*720)

    drawB.rectangle((xmin, ymin, xmax, ymax), fill=255)

    Image.fromarray(np.stack((np.array(maskR), np.array(maskG), np.array(maskB)), axis=-1) ,mode='RGB').save(f'./masks/mask_{row["image"]}', quality = 100)



# %%
