# %%
from operator import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageDraw

# %%

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False, verbose=False)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class BarData:
    def __init__(self, trainres = 256, side = 'outside'):
        self.trainres = trainres
        self.side = side
        self.df = pd.read_csv(f'{DIR_PATH}/training_annotations/vott-csv-export/bar-tracking-export.csv')
        self.df = self.df.sort_values('image')

        self.df = self.df[self.df['label'] == 'outside_plate'].drop('label', axis=1).merge(
            self.df[self.df['label'] != 'outside_plate'].drop('label', axis=1),
            on='image',
            suffixes=['_outer', '_inner']
            ).dropna(axis=0)

        m = len(self.df)
        self.images = torch.empty((m, 3, self.trainres, self.trainres), device='cpu')
        for idx, img in enumerate(self.df['image']):
            img = self.get_image(fn = img)
            self.images[idx] = self.process(img)

        self.outside = torch.tensor(np.array(self.df[['xmin_outer', 'ymin_outer', 'xmax_outer', 'ymax_outer']], dtype=np.float32), device='cpu')
        self.inside = torch.tensor(np.array(self.df[['xmin_inner', 'ymin_inner', 'xmax_inner', 'ymax_inner']], dtype=np.float32), device='cpu')

    def process(self, img, train = True):
        img = img.resize((self.trainres, self.trainres))
        if train:
            img = transforms.RandomGrayscale(0.25)(img)
        img = transforms.ToTensor()(img)
        return img

    def get_image(self, fn:str = None, fp:str = None):
        if fp is None:
            fp = f'{DIR_PATH}/training_images/{fn}'
        return Image.open(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: slice):
        return self.images[idx], self.outside[idx] if self.side=='outside' else self.inside[idx]

class BarModel(nn.Module):
    def __init__(self):
        super(BarModel, self).__init__()
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        x = self.bb(x)
        return x

def test_img(model, epoch, res = 256):
    img = Image.open(f'{DIR_PATH}/../1599942000.jpg')
    model.eval()
    with torch.no_grad():
        tensor = transforms.ToTensor()(img.resize((res,res)))
        tensor = tensor.reshape((1, 3, res, res))
        tensor = tensor.cuda()
        
        bb = model(tensor).tolist()
        id = ImageDraw.Draw(img)

        id.rectangle(bb[0], outline=(0,255,0,0))

    model.train()
    img.save(f'{DIR_PATH}/../image_preds/pred{epoch}.jpg')

def train_epocs(model, optimizer, train_dl, epochs=10):
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y in train_dl:
            batch = y.shape[0]
            x = x.cuda().float()
            y = y.cuda().float()
            out = model(x)
            loss = nn.MSELoss(reduction="mean")(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        print(f'epoch {i} train_loss {train_loss}')
        test_img(model, i)
    return sum_loss/total

def main(side):
    model_path = f'{DIR_PATH}/../models/bar_model_{side}.pth'
    try:
        model = BarModel().cuda()
        parameters = model.parameters() #filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.006)
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print('saved model does not match...')
        model = model.cuda()
        barData = BarData(side = side)
        train_dl = DataLoader(
            barData,
            batch_size=128,
            shuffle=True
        )
        train_epocs(model, optimizer, train_dl, epochs=100)
        torch.save(model.state_dict(), model_path)
    except KeyboardInterrupt:
        print('exiting training loop early...')
        print('saving model...')
        torch.save(model.state_dict(), model_path)
# %%

if __name__ == "__main__":
    main(side = 'inside')
    main(side = 'outside')
else:
    inside = BarModel()
    inside.load_state_dict(torch.load(f'{DIR_PATH}/../models/bar_model_inside.pth'))
    inside = inside.cuda()
    outside = BarModel()
    outside.load_state_dict(torch.load(f'{DIR_PATH}/../models/bar_model_outside.pth'))
    outside = outside.cuda()