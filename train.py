import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

import numpy as np

# import albumentations as A
import timm
import util
from tqdm import tqdm
from dataset import Animal10


util.setSeed(312)

BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
train_tf = T.Compose([
    T.ToTensor(),
    T.Resize((224,224)),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    T.RandomHorizontalFlip(),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    T.Resize((224,224)),
])

train_dataset = Animal10(is_train = True,transform=train_tf)
test_dataset = Animal10(is_train = False,transform=test_tf)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print(timm.list_models('resnet*'))
model = timm.create_model('resnet50',pretrained=True, num_classes = 10).to(device)

optim = torch.optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()


def train():
    for epochs in range(EPOCHS):
        model.train()
        acc = 0
        for (img,label) in tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)

            optim.zero_grad()

            y = model(img)
            loss = criterion(y,label)
            loss.backward()
            optim.step()

            _,y_cls = torch.max(y,1)
            acc += (y_cls == label).sum().item()

        print(f'train acc : {acc}/{len(train_dataset)}')

        model.eval()
        acc = 0
        for (img,label) in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)

            y = model(img)
            loss = criterion(y,label)

            _,y_cls = torch.max(y,1)
            acc += (y_cls == label).sum().item()

        print(f'val acc : {acc}/{len(train_dataset)}')

if __name__ == '__main__':
    train()