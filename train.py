import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import v2
import numpy as np

# import albumentations as A
import timm
import util
from tqdm import tqdm
from dataset import Animal10
from wandblog import Logger
import augmentation
util.setSeed(312)

BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 20
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
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# print(timm.list_models('resnet*'))
target_model = 'resnet18'
model = timm.create_model(target_model,pretrained=True, num_classes = 10).to(device)
optim = torch.optim.Adam(model.parameters())

#baseline기준 CELoss
# criterion = nn.CrossEntropyLoss()
#cutmix기준 multi label 이므로 BCEwithLogitLoss
criterion = nn.BCEWithLogitsLoss()
val_criterion = nn.CrossEntropyLoss()
cut_mix = v2.CutMix(alpha=0.3, num_classes=10)
# sample_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
# grid_cut_mix = augmentation.grid_cut_mix(num_classes=10,max_grid=7, shape=[224,224])
grid_cut_mix = augmentation.grid_cut_mix_v3(num_classes=10,grid=28, shape=[224,224])
wandb_logger = Logger(config = {
    'learning_rate' : LR,
    'architecture' : target_model,
    'batch_size' : BATCH_SIZE,
    'dataset' : "Animal10",
    "epochs" : EPOCHS
})
def train():
    for epochs in range(EPOCHS):
        model.train()
        acc = 0
        for step, (img,label) in enumerate(tqdm(train_loader)):
            # img,label = cut_mix(img,label)
            img,label = grid_cut_mix(img,label)

            img = img.to(device)
            label = label.to(device)
            optim.zero_grad()

            y = model(img)
            loss = criterion(y,label)
            loss.backward()
            optim.step()

            _,y_cls = torch.max(y,1)

            #BCE loss 사용시 
            _,label = torch.max(label,1)

            acc += (y_cls == label).sum().item()

            wandb_logger.log(
                {'data':{'train/loss' : loss.item(), 'train_step': (epochs*len(train_loader)+step)},
                # 'step' : step+(len(train_loader)*epochs)
                })
        # print(f'train acc : {acc}/{len(train_data set)}')
        wandb_logger.log(
            {'data':{'train/acc':acc/len(train_dataset), 'train_epoch': epochs},
            #  'step':epochs+1
            })
        torch.cuda.empty_cache()
        model.eval()    
        acc = 0
        val_loss = 0
        for (img,label) in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)
            with torch.no_grad():
                y = model(img)
                loss = val_criterion(y,label)
            val_loss += loss.item()//len(test_loader)
            _,y_cls = torch.max(y,1)

            acc += (y_cls == label).sum().item()
        wandb_logger.log(
            {'data':{'val/acc':acc/len(test_dataset), 'val/loss':val_loss, 'val_epoch':epochs},
            #  'step':epochs+1
             })  
        torch.cuda.empty_cache()
        # print(f'val acc : {acc}/{len(train_dataset)}')

if __name__ == '__main__':
    train()