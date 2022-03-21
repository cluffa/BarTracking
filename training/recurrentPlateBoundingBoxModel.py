# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageDraw

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False, verbose=False)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class BarData:
    def __init__(self, trainres = 256):
        self.trainres = trainres
        self.df = pd.read_csv(f'{DIR_PATH}/training_annotations/vott-csv-export/bar-tracking-export.csv')
        self.df = self.df.sort_values('image')

        self.df = self.df[self.df['label'] == 'outside_plate'].drop('label', axis=1).merge(
            self.df[self.df['label'] == 'inside_plate'].drop('label', axis=1),
            on='image',
            suffixes=['_outer', '_inner']
            ).dropna(axis=0)

        m = len(self.df)
        self.grayimages = torch.empty((m, 1, self.trainres, self.trainres), device='cpu')
        for idx, img in enumerate(self.df['image']):
            img = self.get_image(fn = img)
            img = self.process(img)
            self.grayimages[idx] = img

        self.outside = torch.tensor(np.array(self.df[['xmin_outer', 'ymin_outer', 'xmax_outer', 'ymax_outer']], dtype=np.float32), device='cpu')
        self.inside = torch.tensor(np.array(self.df[['xmin_inner', 'ymin_inner', 'xmax_inner', 'ymax_inner']], dtype=np.float32), device='cpu')

    def add_rand_pre(self, idx):
        img, (xmin_outer, ymin_outer, xmax_outer, ymax_outer), (xmin_inner, ymin_inner, xmax_inner, ymax_inner) = self.grayimages[idx], self.outside[idx], self.inside[idx]
        h = (ymax_outer - ymin_outer + ymax_inner - ymin_inner)/2
        w = (xmax_outer - xmin_outer + xmax_inner - xmin_inner)/2

        xrand1 = np.random.choice((-1, 1)) * np.random.randint(0, w/4)
        yrand1 = np.random.choice((-1, 1)) * np.random.randint(0, h/4)

        xrand2 = xrand1/2 #+ np.random.randint(-w/24, w/24)
        yrand2 = yrand1/2 #+ np.random.randint(-h/24, h/24)

        max_x = np.max((xmin_outer+xrand1, xmax_outer+xrand1, xmin_inner+xrand1, xmax_inner+xrand1, xmin_outer+xrand2, xmax_outer+xrand2, xmin_inner+xrand2, xmax_inner+xrand2))
        min_x = np.min((xmin_outer+xrand1, xmax_outer+xrand1, xmin_inner+xrand1, xmax_inner+xrand1, xmin_outer+xrand2, xmax_outer+xrand2, xmin_inner+xrand2, xmax_inner+xrand2))
        max_y = np.max((ymin_outer+yrand1, ymax_outer+yrand1, xmin_inner+yrand1, ymax_inner+yrand1, ymin_outer+yrand2, ymax_outer+yrand2, ymin_inner+yrand2, ymax_inner+yrand2))
        min_y = np.min((ymin_outer+yrand1, ymax_outer+yrand1, ymin_inner+yrand1, ymax_inner+yrand1, ymin_outer+yrand2, ymax_outer+yrand2, ymin_inner+yrand2, ymax_inner+yrand2))

        outline1 = Image.new('RGB', (self.trainres, self.trainres), color = (0, 0, 0))
        id1 = ImageDraw.Draw(outline1)
        outline2 = Image.new('RGB', (self.trainres, self.trainres), color = (0, 0, 0))
        id2 = ImageDraw.Draw(outline2)

        id1.rectangle((xmin_outer+xrand1, ymin_outer+yrand1, xmax_outer+xrand1, ymax_outer+yrand1), fill=(127, 127, 127))
        id2.rectangle((xmin_inner+xrand1, ymin_inner+yrand1, xmax_inner+xrand1, ymax_inner+yrand1), fill=(127, 127, 127))
        id1.rectangle((xmin_outer+xrand2, ymin_outer+yrand2, xmax_outer+xrand2, ymax_outer+yrand2), fill=(255, 255, 255))
        id2.rectangle((xmin_inner+xrand2, ymin_inner+yrand2, xmax_inner+xrand2, ymax_inner+yrand2), fill=(255, 255, 255))
        
        outline1 = transforms.Grayscale(1)(outline1)
        outline1 = transforms.ToTensor()(outline1)

        outline2 = transforms.Grayscale(1)(outline2)
        outline2 = transforms.ToTensor()(outline2)

        return torch.cat((outline1, outline2, img))

    def process(self, img):
        img = img.resize((self.trainres, self.trainres))
        img = transforms.Grayscale(1)(img)
        img = transforms.ToTensor()(img)
        return img

    def get_image(self, fn:str = None, fp:str = None):
        if fp is None:
            fp = f'{DIR_PATH}/training_images/{fn}'
        return Image.open(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: slice):
        return self.add_rand_pre(idx), self.outside[idx], self.inside[idx]

class BarModel(nn.Module):
    def __init__(self):
        super(BarModel, self).__init__()
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.outsidebb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 1024), nn.Linear(1024, 4))
        self.insidebb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 1024), nn.Linear(1024, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.outsidebb(x), self.insidebb(x)

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def train_epocs(model, optimizer, train_dl, epochs=10):
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
            loss = loss_out.sum() + loss_in.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
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

# %%
model_path = f'{DIR_PATH}/../models/rnn_bar_model.pth'
train = True
save = True
if __name__ == "__main__" and train:
    try:
        model = BarModel().cuda()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.006)
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print('saved model does not match...')
        model = model.cuda()
        barData = BarData()
        train_dl = DataLoader(
            barData,
            batch_size=128,
            shuffle=True
        )
        train_epocs(model, optimizer, train_dl, epochs=500)
        torch.save(model.state_dict(), model_path)
    except KeyboardInterrupt:
        print('exiting training loop early...')
        print('saving model...')
        torch.save(model.state_dict(), model_path)
    
elif not train:
    model = BarModel().cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    barData = BarData()
else:
    print('doing nothing')

#plt.imshow(torch.mean(barData.add_rand_pre(0), dim = 0))
# %%