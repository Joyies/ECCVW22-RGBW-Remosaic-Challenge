import os
import glob
import random
import pickle

import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import time
import sys
sys.path.append("./") 
from util import ndarray2tensor
import matplotlib.pyplot as plt

def rgb2bayer(rgb):
    # rgb, h*w*3
    h, w, _ = rgb.shape
    bayer = np.zeros((h, w, 1))
    bayer[0:h:2, 0:w:2, 0] = rgb[0:h:2, 0:h:2, 0]
    bayer[0:h:2, 1:w:2, 0] = rgb[0:h:2, 1:h:2, 1]
    bayer[1:h:2, 0:w:2, 0] = rgb[1:h:2, 0:h:2, 1]
    bayer[1:h:2, 1:w:2, 0] = rgb[1:h:2, 1:h:2, 2]

    return bayer

def rgb2qbayer(rgb):
    # rgb, h*w*3
    h, w, _ = rgb.shape
    qbayer = np.zeros((h, w, 1))
    # r
    qbayer[0:h:4, 0:w:4, 0] = rgb[0:h:4, 0:w:4, 0]
    qbayer[0:h:4, 1:w:4, 0] = rgb[0:h:4, 1:w:4, 0]
    qbayer[1:h:4, 0:w:4, 0] = rgb[1:h:4, 0:w:4, 0]
    qbayer[1:h:4, 1:w:4, 0] = rgb[1:h:4, 1:w:4, 0]
    # g
    qbayer[0:h:4, 2:w:4, 0] = rgb[0:h:4, 2:w:4, 1]
    qbayer[0:h:4, 3:w:4, 0] = rgb[0:h:4, 3:w:4, 1]
    qbayer[1:h:4, 2:w:4, 0] = rgb[1:h:4, 2:w:4, 1]
    qbayer[1:h:4, 3:w:4, 0] = rgb[1:h:4, 3:w:4, 1]
    # g
    qbayer[2:h:4, 0:w:4, 0] = rgb[2:h:4, 0:w:4, 1]
    qbayer[2:h:4, 1:w:4, 0] = rgb[2:h:4, 1:w:4, 1]
    qbayer[3:h:4, 0:w:4, 0] = rgb[3:h:4, 0:w:4, 1]
    qbayer[3:h:4, 1:w:4, 0] = rgb[3:h:4, 1:w:4, 1]
    # b
    qbayer[2:h:4, 2:w:4, 0] = rgb[2:h:4, 2:w:4, 2]
    qbayer[2:h:4, 3:w:4, 0] = rgb[2:h:4, 3:w:4, 2]
    qbayer[3:h:4, 2:w:4, 0] = rgb[3:h:4, 2:w:4, 2]
    qbayer[3:h:4, 3:w:4, 0] = rgb[3:h:4, 3:w:4, 2]
    
    return qbayer

def crop_patch(img, patch_size=96, augment=False):
    # crop patch randomly
    h, w, _ = img.shape
    x, y = random.randrange(0, w - patch_size + 1), random.randrange(0, h - patch_size + 1)
    patch = img[y:y+patch_size, x:x+patch_size, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: patch = patch[:, ::-1, :]
        if vflip: patch = patch[::-1, :, :]
        if rot90: patch = patch.transpose(1,0,2)
        # numpy to tensor
    # lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return patch

class DIV2K(data.Dataset):
    def __init__(
        self, HR_folder, 
        train=True, augment=True,
        patch_size=96, repeat=168
    ):
        super(DIV2K, self).__init__()
        self.HR_folder = HR_folder
        self.augment   = augment
        self.img_postfix = '.png'
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train

        ## for raw png images
        self.img_files = {}

        ## generate dataset
        if self.train:
            self.start_idx = 1
            self.end_idx = 791
        else:
            self.start_idx = 791
            self.end_idx = 801

        image_dir_list = os.listdir(HR_folder)  
        for i in range(self.start_idx, self.end_idx):
            if (i+1) % 20 == 0:
                print("read {} hr filenames!".format(i+1))
            self.img_files[str(i)] = []
            idx = str(i).zfill(4)
            for image_dir in image_dir_list:
                if os.path.basename(image_dir)[0:4] == idx:  # 
                    hr_image_file = os.path.join(HR_folder, image_dir)
                    self.img_files[str(i)].append(hr_image_file)

        LEN = self.end_idx - self.start_idx
        self.nums_trainset = LEN
        
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        img_patchnames = self.img_files[str(idx+1)]
        patch_idx = np.random.randint(0, len(img_patchnames))
        img = img_patchnames[patch_idx]
        img = imageio.imread(img, pilmode="RGB")
        if self.train:
            img = crop_patch(img, self.patch_size, True)
            # rgb2bayer, rg2qbayer
            bayer, qbayer = ndarray2tensor(rgb2bayer(img)), ndarray2tensor(rgb2qbayer(img))
            batch = {}
            batch['bayer_sg'] = bayer
            batch['qbayer_sg'] = qbayer
            return batch

        bayer, qbayer = ndarray2tensor(rgb2bayer(img)), ndarray2tensor(rgb2qbayer(img))
        batch = {}
        batch['bayer_sg'] = bayer
        batch['qbayer_sg'] = qbayer
        return batch