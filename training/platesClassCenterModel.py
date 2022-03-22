# %%
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

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False, verbose=False)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class PlatesData:
    def __init__(self):
        # 'image','xmin','ymin','xmax','ymax','label'
        self.df = pd.read_csv(f'{DIR_PATH}/plate_images/plate-center-export.csv').drop(['xmax','ymax'], axis=1)
        self.df['label'] = self.df['image'].str.split('-', expand=True)[0]
        m = len(self.df)
        self.images = torch.empty((m, 3, 128, 128), device='cpu')
        for idx, img in enumerate(self.df['image']):
            img = self.get_image(img)
            self.images[idx] = self.process(img)

        self.center = torch.tensor(np.array(self.df[['xmin', 'ymin']], dtype=np.float32), device='cpu')
        self.labels = torch.tensor(np.array(pd.get_dummies(self.df['label']), dtype=np.float32), device='cpu')

    def process(self, img, train = True):
        if train:
            img = transforms.RandomGrayscale(0.5)(img)
        img = transforms.ToTensor()(img)
        return img

    # cords are center of radius 5 red square
    def get_image(self, im, cords: tuple = None, realCenter: bool = False, fp:str = None):
        if fp is None:
            if isinstance(im, int):
                fn = self.df['image'][im]
            else:
                fn = im
            if realCenter:
                cords = (self.df['xmin'][im], self.df['ymin'][im])
            
            fp = f'{DIR_PATH}/plate_images/images/{fn}'

        if cords is None:
            return Image.open(fp)
        else:
            img = Image.open(fp)
            id = ImageDraw.Draw(img)
            id.ellipse((cords[0]-5, cords[1]-5, cords[0]+5, cords[1]+5), fill=(255,0,0,0))
            return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: slice):
        return self.images[idx], self.labels[idx], self.center[idx]

class PlateModel(nn.Module):
    def __init__(self):
        super(PlateModel, self).__init__()
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 2))
        self.center = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 2))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.center(x)

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def train_epocs(model, optimizer, train_dl, epochs=10,C=100):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb in train_dl:
            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.mse_loss(out_bb, y_bb, reduction="sum")
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        print(f'epoch {i} train_loss {train_loss}')
    return sum_loss/total
# %%

def get_prediction(image: Image = None, fp: str = None, tensor: torch.Tensor = None):
    model.eval()
    with torch.no_grad():
        if fp is not None:
            image = Image.open(fp)
            image = image.resize((128, 128))
            tensor = plate_data.process(image)
        elif image is not None:
            image = image.resize((128, 128))
            tensor = plate_data.process(image)

        if (len(tensor.shape) < 4):
            tensor = tensor.reshape((1, 3, 128, 128))
        label, center = model(tensor.cuda())
        label, center = label.cpu().numpy()[0], center.cpu().numpy()[0]

        return label, center

def add_points(image: Image = None, fp: str = None, center = None, label = None):
    if fp is not None and image is None:
        image = Image.open(fp)
    
    image = image.resize((128, 128))

    if center is None or label is None:
        predlabel, (x, y) = get_prediction(image=image)

    label = 'inside' if predlabel[0] > predlabel[1] else 'outside'

    draw = ImageDraw.Draw(image)
    draw.ellipse((x-3, y-3, x+3, y+3), fill=(255,0,0,0))
    draw.text((0., 0.), text = label)
    return image

override = False
if __name__ == "__main__" and not override:
    model = PlateModel().cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.006)
    plate_data = PlatesData()
    train_dl = DataLoader(
        plate_data,
        batch_size=512,
        shuffle=True
    )
    try:
        model.load_state_dict(torch.load(f'{DIR_PATH}/../models/plate_model.pth'))
        model = model.cuda()
        train_epocs(model, optimizer, train_dl, epochs=250)
    except KeyboardInterrupt:
        pass
    torch.save(model.state_dict(), f'{DIR_PATH}/../models/plate_model.pth')
else:
    model = PlateModel().cuda()
    model.load_state_dict(torch.load(f'{DIR_PATH}/../models/plate_model.pth'))
    model = model.cuda()