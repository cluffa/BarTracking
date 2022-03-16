# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from PIL import Image


real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

# %%
class Data:
    def __init__(self, color: bool, res: int, label_filter: str) -> None:
        colorStr = 'RGB' if color else 'GRAY'
        with open(f'{dir_path}/annotations{colorStr}{res}.json') as f:
            data = json.load(f)
    
        data = pd.DataFrame.from_records(data)
        data = data[data['label'] == label_filter]

        self.images = np.empty((len(data['id']), 3 if color else 1, res, res))
        for idx, id in enumerate(data['id']):
            path = f'{dir_path}/images{colorStr}{res}/{id}.jpg'
            self.images[idx] = np.array(Image.open(path)).reshape((3 if color else 1, res, res))

        self.bounding_boxes = np.array(data[['row', 'column', 'height', 'width']])

        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.bounding_boxes = torch.tensor(self.bounding_boxes, dtype=torch.float32)
        

    def __len__(self):
        return len(self.bounding_boxes)

    def __getitem__(self, idx):
        return self.images[idx], self.bounding_boxes[idx]

test_data = Data(True, 480, 'outside_plate')[0:4]
test_images, test_boxes = test_data
# %%

class NN(nn.Module):
    def __init__(self, res):
        super(NN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((128,128))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (16, 16)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((128,128))
        )

        self.drop = nn.Dropout(0.1)
        self.flattenMany = nn.Flatten(1)
        self.flattenSingle = nn.Flatten(0)

        self.dense1 = nn.Sequential(
            nn.Linear(524288, 512),
            nn.ReLU()
        )
        self.dense2 = nn.Sequential(
            nn.Linear(512, 4),
            
        )

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        #print(x.shape)
        #x = self.conv2(x)
        #print(x.shape)
        x = self.drop(x)

        # allows allows for missing n, single input
        if len(x.shape) == 3:
            x = self.flattenSingle(x)
        else:
            x = self.flattenMany(x)
        
        #print(x.shape)

        x = self.dense1(x)
        #print(x.shape)
        x = self.dense2(x)

        return x

model = NN(480).cuda()
model(test_images.cuda())
# %%

loss_fn = nn.L1Loss().cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
def train(epochs, num_batches):
    model.train()
    dataloader = DataLoader(
        Data(True, 480, 'outside_plate'),
        batch_size=num_batches,
        shuffle=True
    )

    for epoch in range(epochs):
        losses = []
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(epoch, np.mean(losses))


# %%

if False:
    model.load_state_dict(torch.load('./model.pth'))
    model = model.cuda()
else:
    train(10, 10)
    torch.save(model.state_dict(), './model.pth')

# %%

def predict(model, image):
    model.eval()
    with torch.no_grad():
        pred = model(image.cuda())
    return pred

print(predict(model, test_images[0]))

def plot_boxes(image, box):
    pass
# %%
