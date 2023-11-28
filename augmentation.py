import torch
import numpy as np
from itertools import cycle

class grid_cut_mix():
    #grid_type is in ['grid','horizontal', 'vertical']
    #현재 버전은 dataset 이미지의 해상도가 동일할때 적용 가능
    def __init__(self, num_classes,shape:list,sample_loader, p=0.3,grid=3,grid_type='grid'):
        self.p = p
        self.grid = grid
        self.grid_type = grid_type
        self.num_classes = num_classes
        self.shape=shape
        if len(shape) != 2:
            raise ValueError("2차원 배열만 사용 가능합니다.")
        horizontal = torch.repeat_interleave(torch.Tensor([True,False]).bool(),grid)
        horizontal = horizontal.repeat((shape[1]+2*grid-1)//(2*grid))[:shape[1]].view(1,-1)
        vertical = torch.repeat_interleave(torch.Tensor([True,False]).bool(),grid)
        vertical = vertical.repeat((shape[1]+2*grid-1)//(2*grid))[:shape[1]].view(-1,1)
        slice_map = torch.zeros(shape,dtype=torch.bool)
        self.sample_loader = cycle(sample_loader)
        if grid_type == 'horizontal':
            slice_map[horizontal[0],:]=True
        elif grid_type == 'vertical':
            slice_map[:,vertical[:,0]] = True
        elif grid_type == 'grid':
            slice_map = horizontal*vertical + (~horizontal)*(~vertical)
        self.slice_map = slice_map
    def __call__(self,img,label):
        label = torch.nn.functional.one_hot(label,self.num_classes).float()
        p = torch.rand(img.shape[0]) < self.p
        sample_img,sample_label = next(self.sample_loader)
        sample_label = torch.nn.functional.one_hot(sample_label,self.num_classes).float()
        new_sample = img[p]
        target_sample = sample_img[p]
        new_sample[:,:,self.slice_map] = target_sample[:,:,self.slice_map]
        img[p] = new_sample
        label[p] = (label[p]+sample_label[p])/2
        return img,label

class grid_cut_mix_v2():
    #grid_type is in ['grid','horizontal', 'vertical']
    #현재 버전은 dataset 이미지의 해상도가 동일할때 적용 가능
    #v2. p 파라메터 제거, grid cutmix 알고리즘 변경, 배치 내에서 grid cutmix 진행
    def __init__(self, num_classes,shape:list,max_grid=6,grid_type='grid'):
        self.max_grid = max_grid
        self.grid_type = grid_type
        self.num_classes = num_classes
        self.shape=shape
        if len(shape) != 2:
            raise ValueError("2차원 배열만 사용 가능합니다.")
        if grid_type not in ['grid','horizontal','vertical']:
            raise ValueError("grid type이 허용되지 않은 종류입니다.")
    def __call__(self,img,label):
        #1/2~ 1/4비율로 섞기
        ratio = np.random.randint(2,4)
        grid = np.random.randint(1,self.max_grid+1)
        true_idx = np.random.randint(ratio)
        if self.grid_type =='horizontal':
            horizontal = torch.zeros(ratio).bool()
            horizontal[true_idx] = True
            horizontal = horizontal.repeat_interleave(grid)
            horizontal = horizontal.repeat((self.shape[1]+2*grid-1)//(2*grid))[:self.shape[1]].view(-1,1).repeat(1,self.shape[1])
            slice_map = horizontal
        elif self.grid_type == 'vertical':
            vertical = torch.zeros(ratio).bool()
            vertical[true_idx] = True
            vertical = vertical.repeat_interleave(grid)
            vertical = vertical.repeat((self.shape[1]+2*grid-1)//(2*grid))[:self.shape[1]].view(1,-1).repeat(self.shape[0],1)
            slice_map = vertical
        elif self.grid_type == 'grid':
            slice_map = torch.zeros(ratio*ratio).bool()
            rand_true = torch.randperm(ratio*ratio)[:ratio]
            slice_map[rand_true] = True
            slice_map = slice_map.reshape(ratio,ratio)
            slice_map = slice_map.repeat_interleave(grid,1).repeat_interleave(grid,0)
            slice_map = slice_map.repeat((self.shape[1]+2*grid-1)//(2*grid),(self.shape[1]+2*grid-1)//(2*grid))[:self.shape[0],:self.shape[1]]
        slice_map = slice_map.repeat(3,1,1)
        shuffle_idx = torch.randperm(img.shape[0])
        label = torch.nn.functional.one_hot(label,self.num_classes).float()
        shuffle_label = label.clone()
        shuffle_label = shuffle_label/ratio
        shuffle_label += label[shuffle_idx]*(1-1/ratio)
        shuffle_img = img.clone()
        shuffle_img[:,slice_map] = img[shuffle_idx][:,slice_map]
        return shuffle_img, shuffle_label
    
class grid_cut_mix_v3():
    #grid_type is in ['grid','horizontal', 'vertical']
    #현재 버전은 dataset 이미지의 해상도가 동일할때 적용 가능
    #v3. p 파라메터 제거, random scale grid 제거
    def __init__(self, num_classes,shape:list,grid=6,grid_type='grid'):
        self.grid = grid
        self.grid_type = grid_type
        self.num_classes = num_classes
        self.shape=shape
        if len(shape) != 2:
            raise ValueError("2차원 배열만 사용 가능합니다.")
        if grid_type not in ['grid','horizontal','vertical']:
            raise ValueError("grid type이 허용되지 않은 종류입니다.")
    def __call__(self,img,label):
        #1/2~ 1/4비율로 섞기
        ratio = np.random.randint(2,4)
        true_idx = np.random.randint(ratio)
        grid = self.grid
        if self.grid_type =='horizontal' :
            horizontal = torch.zeros(ratio).bool()
            horizontal[true_idx] = True
            horizontal = horizontal.repeat_interleave(grid)
            horizontal = horizontal.repeat((self.shape[1]+2*grid-1)//(2*grid))[:self.shape[1]].view(-1,1).repeat(1,self.shape[1])
            slice_map = horizontal
        elif self.grid_type == 'vertical':
            vertical = torch.zeros(ratio).bool()
            vertical[true_idx] = True
            vertical = vertical.repeat_interleave(grid)
            vertical = vertical.repeat((self.shape[1]+2*grid-1)//(2*grid))[:self.shape[1]].view(1,-1).repeat(self.shape[0],1)
            slice_map = vertical
        elif self.grid_type == 'grid':
            slice_map = torch.zeros(ratio*ratio).bool()
            rand_true = torch.randperm(ratio*ratio)[:ratio]
            slice_map[rand_true] = True
            slice_map = slice_map.reshape(ratio,ratio)
            slice_map = slice_map.repeat_interleave(grid,1).repeat_interleave(grid,0)
            slice_map = slice_map.repeat((self.shape[1]+2*grid-1)//(2*grid),(self.shape[1]+2*grid-1)//(2*grid))[:self.shape[0],:self.shape[1]]
        slice_map = slice_map.repeat(3,1,1)
        shuffle_idx = torch.randperm(img.shape[0])
        label = torch.nn.functional.one_hot(label,self.num_classes).float()
        shuffle_label = label.clone()
        shuffle_label = shuffle_label/ratio
        shuffle_label += label[shuffle_idx]*(1-1/ratio)
        shuffle_img = img.clone()
        shuffle_img[:,slice_map] = img[shuffle_idx][:,slice_map]
        return shuffle_img, shuffle_label