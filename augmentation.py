import torch
import numpy as np
from itertools import cycle

class grid_cut_mix():
    #grid_type is in ['grid','horizontal', 'vertical']
    #현재 버전은 dataset 이미지의 해상도가 동일할때 적용 가능
    def __init__(self, num_classes,shape:list,sample_loader, alpha=0.3,grid=3,grid_type='grid'):
        self.alpha = alpha
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
        p = torch.rand(img.shape[0]) < self.alpha
        sample_img,sample_label = next(self.sample_loader)
        sample_label = torch.nn.functional.one_hot(sample_label,self.num_classes).float()
        new_sample = img[p]
        target_sample = sample_img[p]
        new_sample[:,:,self.slice_map] = target_sample[:,:,self.slice_map]
        img[p] = new_sample
        label[p] = (label[p]+sample_label[p])/2
        return img,label