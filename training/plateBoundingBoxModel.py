# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from IPython import display

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False, verbose=False)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class BarData:
    def __init__(self):
        self.df = pd.read_csv(f'{DIR_PATH}/training_annotations/vott-csv-export/bar-tracking-export.csv')
        self.df = self.df.sort_values('image')

        self.df = self.df[self.df['label'] == 'outside_plate'].drop('label', axis=1).merge(
            self.df[self.df['label'] != 'outside_plate'].drop('label', axis=1),
            on='image',
            suffixes=['_outer', '_inner']
            ).dropna(axis=0)

        m = len(self.df)
        self.images = torch.empty((m, 3, 720, 720), device='cpu')
        for idx, img in enumerate(self.df['image']):
            img = self.get_image(fn = img)
            self.images[idx] = self.process(img)

        self.outside = torch.tensor(np.array(self.df[['xmin_outer', 'ymin_outer', 'xmax_outer', 'ymax_outer']], dtype=np.float32), device='cpu')
        self.inside = torch.tensor(np.array(self.df[['xmin_inner', 'ymin_inner', 'xmax_inner', 'ymax_inner']], dtype=np.float32), device='cpu')

    def process(self, img, train = True):
        if train:
            img = transforms.RandomGrayscale(0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img

    def get_image(self, fn:str = None, fp:str = None):
        if fp is None:
            fp = f'{DIR_PATH}/training_images/{fn}'
        return Image.open(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: slice):
        return self.images[idx], self.outside[idx], self.inside[idx]

#barData = BarData()
# test = barData[:4]
#print('test shapes for 4 samples:',*[i.shape for i in test])
# %%
# train_dl = DataLoader(
#     barData,
#     batch_size=16,
#     shuffle=True
# )
# valid_dl = DataLoader(
#     barData,
#     batch_size=128,
#     shuffle=True
# )

class BarModel(nn.Module):
    def __init__(self):
        super(BarModel, self).__init__()
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:7])
        self.features2out = nn.Sequential(*layers[7:])
        self.features2in = nn.Sequential(*layers[7:])
        self.outsidebb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.insidebb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        #shared
        x = self.features1(x)

        #split
        z = self.features2in(x)
        x = self.features2out(x)

        #outside
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        x = self.outsidebb(x)

        #inside
        z = F.relu(z)
        z = nn.AdaptiveAvgPool2d((1,1))(z)
        z = z.view(z.shape[0], -1)
        z = self.insidebb(z)

        return x, z

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

# def val_metrics(model, valid_dl, C=1000):
#     model.eval()
#     total = 0
#     sum_loss = 0
#     #correct = 0 
#     for x, y_out, y_in in valid_dl:
#         batch = y_class.shape[0]
#         x = x.cuda().float()
#         y_class = y_class.cuda()
#         y_bb = y_bb.cuda().float()
#         out_class, out_bb = model(x)
#         loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
#         loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
#         loss_bb = loss_bb.sum()
#         loss = loss_class + loss_bb/C
#         _, pred = torch.max(out_class, 1)
#         #correct += pred.eq(y_class).sum().item()
#         sum_loss += loss.item()
#         total += batch
#     return sum_loss/total#, correct/total

def train_epocs(model, optimizer, train_dl, epochs=10):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_out, y_in in train_dl:
            batch = y_out.shape[0]
            x = x.cuda().float()
            y_out = y_out.cuda().float()
            y_in = y_in.cuda().float()
            out_out, out_in = model(x)
            loss_out = F.l1_loss(out_out, y_out, reduction="none").sum(1)
            loss_in = F.l1_loss(out_in, y_in, reduction="none").sum(1)
            loss_out = loss_out.sum()
            loss_in = loss_in.sum()
            loss = loss_in + loss_out
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        #val_loss, val_acc = val_metrics(model, valid_dl, C)
        #val_loss = val_metrics(model, valid_dl, C)
        #print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))
        print(f'epoch {i} train_loss {train_loss}')
    return sum_loss/total
# %%


# def get_prediction(image: Image = None, fp: str = None, tensor: torch.Tensor = None):
#     model.eval()
#     with torch.no_grad():
#         if fp is not None:
#             image = Image.open(fp)
#             image = image.resize((128, 128))
#             tensor = plate_data.process(image)
#         elif image is not None:
#             image = image.resize((128, 128))
#             tensor = plate_data.process(image)

#         if (len(tensor.shape) < 4):
#             tensor = tensor.reshape((1, 3, 128, 128))
#         label, center = model(tensor.cuda())
#         label, center = label.cpu().numpy()[0], center.cpu().numpy()[0]

#         return label, center

# def add_points(image: Image = None, fp: str = None, center = None, label = None):
#     if fp is not None and image is None:
#         image = Image.open(fp)
    
#     image = image.resize((128, 128))

#     if center is None or label is None:
#         predlabel, (x, y) = get_prediction(image=image)
    
#     #x, y = x/2, y/2

#     label = 'inside' if predlabel[0] > predlabel[1] else 'outside'
#     #image = image.resize((512, 512))

#     draw = ImageDraw.Draw(image)
#     draw.ellipse((x-3, y-3, x+3, y+3), fill=(255,0,0,0))
#     draw.text((0., 0.), text = label)
#     return image

model_path = f'{DIR_PATH}/../models/bar_model.pth'
override = False
if __name__ == "__main__" and not override:
    try:
        model = BarModel().cuda()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.006)
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        barData = BarData()
        train_dl = DataLoader(
            barData,
            batch_size=16,
            shuffle=True
        )
        train_epocs(model, optimizer, train_dl, epochs=100)
    except KeyboardInterrupt:
        pass
    torch.save(model.state_dict(), model_path)
else:
    model = BarModel().cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()