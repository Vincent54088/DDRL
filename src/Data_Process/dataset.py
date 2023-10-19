import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import random

class DataLoaderTrain(Dataset):
    def __init__(self, data_path, patch_size,mode = 'train'):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(data_path, 'multiScences')))
        tar_files = sorted(os.listdir(os.path.join(data_path, 'gtScences')))

        self.inp_filenames = [os.path.join(data_path, 'multiScences', x)  for x in inp_files]
        self.tar_filenames = [os.path.join(data_path, 'gtScences', x) for x in tar_files]

        self.sizex = len(self.tar_filenames)  # get the size of target
        print(self.sizex)
        self.ps = patch_size
        self.mode = mode

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        ps = self.ps

        inp_path = self.inp_filenames[index]
        tar_path = self.tar_filenames[index]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # pair
        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]


        rr  = random.randint(0, hh-ps)
        cc  = random.randint(0, ww-ps)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # horizontal flip
        if self.mode == 'train' and random.randint(0, 1) == 1:
            inp_img = torch.flip(inp_img, dims=[2])
            tar_img = torch.flip(tar_img, dims=[2])

        inp_img = inp_img*2 - 1
        tar_img = tar_img*2 - 1

        return inp_img,tar_img,tar_path

class Mydataset(Dataset):
    def __init__(self,free_foggy_images_path:list,foggy_images_path:list,train = True):
        super(Mydataset,self).__init__()
        self.free_foggy_img_list = free_foggy_images_path
        self.foggy_img_list = foggy_images_path
        self.train = train

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        tar_img = Image.open(self.free_foggy_img_list[index]).convert('RGB')
        inp_img = Image.open(self.foggy_img_list[index]).convert('RGB')
        W,H = inp_img.size
        
        w = W%8
        h = H%8

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        inp_img = inp_img[:, :H-h, :W-w]
        tar_img = tar_img[:, :H-h, :W-w]

        inp_img = inp_img*2 - 1
        tar_img = tar_img*2 - 1
        return inp_img,tar_img

    def __len__(self):
        return len(self.foggy_img_list)

def load_data(img_path):
    '''
    param img_path : the directory of free_foggy_image or foggy_image
    return image_list and depth_list
    '''
    img_path_list = []
    for root,_,fnames in os.walk(img_path):
        for fname in fnames:
            path = os.path.join(root,fname)
            img_path_list.append(path)
    return img_path_list

#get datasets
def get_dataset(foggy_imgFile,free_foggy_imgFile,train = True):
    foggy_img = load_data(foggy_imgFile)
    free_foggy_img = load_data(free_foggy_imgFile)
    dataset = Mydataset(free_foggy_img,foggy_img,train=train)
    return dataset

