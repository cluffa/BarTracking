# %%
import pandas as pd
from PIL import Image, ImageDraw

# %%
# import annotations.csv
annotations = pd.read_csv('./train_data/annotations.csv')
for i in range(len(annotations)):
    row = annotations.iloc[i]
    img = Image.open(f'./train_data/images/{row["image"]}')

    # create empty rgb mask image
    mask = Image.new('RGB', img.size, (0, 0, 0))

    # draw ellipse, red is inside, green is outside
    draw = ImageDraw.Draw(mask)
    draw.ellipse((row['xmin_inside_plate']*720, row['ymin_inside_plate']*720, row['xmax_inside_plate']*720, row['ymax_inside_plate']*720), fill=(255, 0, 0))
    draw.ellipse((row['xmin_outside_plate']*720, row['ymin_outside_plate']*720, row['xmax_outside_plate']*720, row['ymax_outside_plate']*720), fill=(0, 255, 0))

    mask.save(f'./train_data/masks/mask_{row["image"]}', quality = 100)


