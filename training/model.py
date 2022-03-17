# %%
import imp
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

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

RES = 720
LABEL = 'inside_plate'
COLOR = True # doesn't work with grayscale yet

# %%

class Data:
    def __init__(self, color: bool = COLOR, res: int = RES, label_filter: str = LABEL) -> None:
        colorStr = 'RGB' if color else 'GRAY'
        with open(f'{DIR_PATH}/annotations{colorStr}{res}.json') as f:
            data = json.load(f)

        data = pd.DataFrame.from_records(data)
        data = data.sort_values('id')
        data = data[data['label'] == label_filter]
        data['row_end'] = data['row'] + data['height']
        data['column_end'] = data['column'] + data['width']
        
        self.images = []
        for id in data['id']:
            path = f'{DIR_PATH}/images{colorStr}{res}/{id}.jpg'
            self.images.append(Image.open(path))

        self.bb_array = np.array(data[['column', 'row', 'column_end', 'row_end']])

        self.x_preprocess_train = transforms.Compose([
            transforms.RandomAutocontrast(p = 0.25),
            transforms.RandomInvert(p = 0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p = 0.25),
        ])

        self.x_preprocess_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        multiply = 1
        img_stacks = []
        for i in range(multiply):
            img_stacks.append(torch.stack([self.x_preprocess_test(image) for image in self.images]))

        self.images_tensor = torch.cat(img_stacks).to(dtype=torch.float32).cpu()
        self.bb = torch.tensor(self.bb_array)
        self.bb = torch.cat([self.bb]*multiply).to(dtype=torch.float32).cpu()

        self.images_array = [np.array(image, dtype=np.int32) for image in self.images]
        self.images_array = np.stack(self.images_array)

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, idx):
        #return self.images_tensor[idx], self.bb[idx]
        return self.x_preprocess_train(self.images[idx]).to(dtype=torch.float32).cpu(), self.bb[idx]

# %%
class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        #self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.bb(x)

# %%
# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, (10, 10)),
#             nn.ReLU()
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 1, (3, 3)),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((720, 720))
#         )

#         self.drop = nn.Dropout(0.1)

#     def forward(self, x:torch.Tensor):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x


# %%
model = BB_model().cuda()
loss_fn = nn.L1Loss().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optim = torch.optim.Adam(parameters, lr=0.005)


# %%
def train(epochs, num_batches):
    model.train()
    dataloader = DataLoader(
        Data(),
        batch_size=num_batches,
        shuffle=True
    )
    
    mean_losses = []
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

        mean_losses.append(np.mean(losses))
        print(epoch, mean_losses[-1], end = '')
        if epoch > 50:
            if mean_losses[-1] < np.min(mean_losses[:-1]):
                torch.save(model.state_dict(), f'{DIR_PATH}/checkpoints/model.pth')
                print(' saved')
            else:
                print('')
        else:
            print('')


# %%
# True to load trained model
# from collections import OrderedDict

# def convert(lst):
#     if isinstance(lst, list):
#         for idx, element in enumerate(lst):
#             lst[idx] = convert(element)
#     elif isinstance(lst, OrderedDict) or isinstance(lst, dict):
#         for key in lst.keys():
#             lst[key] = convert(lst[key])
#     elif isinstance(lst, torch.Tensor):
#         lst = convert(lst.tolist())
#     return lst

if True:
    #model.load_state_dict(torch.load(f'{DIR_PATH}/checkpoints/model.pth'))
    model.load_state_dict(torch.load(f'{DIR_PATH}/model_finished1.pth'))
    model = model.cuda()
    # od = model.state_dict()
    # od = convert(od)

    # with open("model.json", "w") as f:
    #     json.dump(json.dumps(od, indent=4), f)

else:
    train(500, 16)
    torch.save(model.state_dict(), f'{DIR_PATH}/model_finished.pth')

# %%

def predict(model, image):
    model.eval()
    with torch.no_grad():
        pred = model(image.cuda())
    return pred

j = 10
test_images = Data().images_tensor[0:j]
pred = predict(model, test_images).cpu().numpy()
pred = pred.astype(np.int32)
print(pred)
for i in range(j):
    
    img = Data().images_array[i]
    if np.all(pred[i] < 720) and np.all(pred[i] >= 0):
        img[pred[i, 1], :, :] = 255
        img[:, pred[i, 0], :] = 255
        img[pred[i, 3], :, :] = 255
        img[:, pred[i, 2], :] = 255
        
    plt.imshow(img)
    plt.show()

# %%

# %%
