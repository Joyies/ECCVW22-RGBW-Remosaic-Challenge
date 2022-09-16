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
from util import ndarray2tensor, read_bin_file
import matplotlib.pyplot as plt


def rggb2bayer(rggb):
    h, w, _ = rggb.shape
    bayer = np.zeros((2*h, 2*w))
    bayer[0:2*h:2, 0:2*w:2] = rggb[:, :, 0]
    bayer[0:2*h:2, 1:2*w:2] = rggb[:, :, 1]
    bayer[1:2*h:2, 0:2*w:2] = rggb[:, :, 2] 
    bayer[1:2*h:2, 1:2*w:2] = rggb[:, :, 3]

    return bayer

def rggb2qbayer(rggb):
    h, w, _ = rggb.shape
    qbayer = np.zeros((2*h, 2*w))
    # r
    qbayer[0:2*h:4, 0:2*w:4] = rggb[0:h:2, 0:w:2, 0]
    qbayer[0:2*h:4, 1:2*w:4] = rggb[0:h:2, 1:w:2, 0]
    qbayer[1:2*h:4, 0:2*w:4] = rggb[1:h:2, 0:w:2, 0] 
    qbayer[1:2*h:4, 1:2*w:4] = rggb[1:h:2, 1:w:2, 0] 
    # g
    qbayer[0:2*h:4, 2:2*w:4] = rggb[0:h:2, 0:w:2, 1] 
    qbayer[0:2*h:4, 3:2*w:4] = rggb[0:h:2, 1:w:2, 1] 
    qbayer[1:2*h:4, 2:2*w:4] = rggb[1:h:2, 0:w:2, 1]
    qbayer[1:2*h:4, 3:2*w:4] = rggb[1:h:2, 1:w:2, 1]

    qbayer[2:2*h:4, 0:2*w:4] = rggb[0:h:2, 0:w:2, 2]
    qbayer[2:2*h:4, 1:2*w:4] = rggb[0:h:2, 1:w:2, 2]
    qbayer[3:2*h:4, 0:2*w:4] = rggb[1:h:2, 0:w:2, 2] 
    qbayer[3:2*h:4, 1:2*w:4] = rggb[1:h:2, 1:w:2, 2]

    qbayer[2:2*h:4, 2:2*w:4] = rggb[0:h:2, 0:w:2, 3]
    qbayer[2:2*h:4, 3:2*w:4] = rggb[0:h:2, 1:w:2, 3] 
    qbayer[3:2*h:4, 2:2*w:4] = rggb[1:h:2, 0:w:2, 3] 
    qbayer[3:2*h:4, 3:2*w:4] = rggb[1:h:2, 1:w:2, 3] 

    return qbayer

def bayer2rggb(bayer):
    h, w = bayer.shape
    rggb = np.zeros((h // 2, w // 2, 4))
    rggb[:, :, 0] = bayer[0:h:2, 0:w:2]
    rggb[:, :, 1] = bayer[0:h:2, 1:w:2]
    rggb[:, :, 2] = bayer[1:h:2, 0:w:2]
    rggb[:, :, 3] = bayer[1:h:2, 1:w:2]

    return rggb


def qbayer2rggb(qbayer):
    h, w = qbayer.shape
    rggb = np.zeros((h // 2, w // 2, 4))
    # r
    rggb[0:h//2:2, 0:w//2:2, 0] = qbayer[0:h:4, 0:w:4]
    rggb[0:h//2:2, 1:w//2:2, 0] = qbayer[0:h:4, 1:w:4]
    rggb[1:h//2:2, 0:w//2:2, 0] = qbayer[1:h:4, 0:w:4]
    rggb[1:h//2:2, 1:w//2:2, 0] = qbayer[1:h:4, 1:w:4]
    # g
    rggb[0:h//2:2, 0:w//2:2, 1] = qbayer[0:h:4, 2:w:4]
    rggb[0:h//2:2, 1:w//2:2, 1] = qbayer[0:h:4, 3:w:4]
    rggb[1:h//2:2, 0:w//2:2, 1] = qbayer[1:h:4, 2:w:4]
    rggb[1:h//2:2, 1:w//2:2, 1] = qbayer[1:h:4, 3:w:4]

    rggb[0:h//2:2, 0:w//2:2, 2] = qbayer[2:h:4, 0:w:4]
    rggb[0:h//2:2, 1:w//2:2, 2] = qbayer[2:h:4, 1:w:4]
    rggb[1:h//2:2, 0:w//2:2, 2] = qbayer[3:h:4, 0:w:4]
    rggb[1:h//2:2, 1:w//2:2, 2] = qbayer[3:h:4, 1:w:4]

    rggb[0:h//2:2, 0:w//2:2, 3] = qbayer[2:h:4, 2:w:4]
    rggb[0:h//2:2, 1:w//2:2, 3] = qbayer[2:h:4, 3:w:4]
    rggb[1:h//2:2, 0:w//2:2, 3] = qbayer[3:h:4, 2:w:4]
    rggb[1:h//2:2, 1:w//2:2, 3] = qbayer[3:h:4, 3:w:4]

    return rggb
    
def crop_patch(bayer, qbayer, patch_size, augment=True):
    # crop patch randomly
    lh, lw, _ = bayer.shape
    lx, ly = random.randrange(0, lw - patch_size + 1, 2), random.randrange(0, lh - patch_size + 1, 2)
    bayer, qbayer = bayer[ly:ly+patch_size, lx:lx+patch_size, :], qbayer[ly:ly+patch_size, lx:lx+patch_size, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: bayer, qbayer = bayer[:, ::-1, :], qbayer[:, ::-1, :]
        if vflip: bayer, qbayer = bayer[::-1, :, :], qbayer[::-1, :, :]
        if rot90: bayer, qbayer = bayer.transpose(1,0,2), qbayer.transpose(1,0,2)
        # numpy to tensor
    bayer, qbayer = ndarray2tensor(bayer), ndarray2tensor(qbayer)
    return bayer, qbayer

class Remosaic(data.Dataset):
    def __init__(
        self, B_folder, QB_0db_folder, QB_24db_folder, QB_42db_folder, 
        train=True, augment=True, 
        patch_size=96, repeat=168
    ):
        super(Remosaic, self).__init__()
        self.B_folder = B_folder
        self.augment   = augment
        self.img_postfix = '.bin'
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train

        ## for raw files
        self.b_files = []
        self.qb_0db_files = []
        self.qb_24db_files = []
        self.qb_42db_files = []
        ## for raw png images
        self.b_images = []
        self.qb_0db_images = []
        self.qb_24db_images = []
        self.qb_42db_images = []

        image_dir_list = os.listdir(B_folder)  
        for i in range(len(image_dir_list)):
            image_dir = image_dir_list[i]
            if (i+1) % 20 == 0:
                print("read {} hr filenames!".format(i+1))  
            b_image_dir = image_dir_list[i]
            qb_0db_image_dir = b_image_dir.split('.')[0] + '_0db.bin'
            qb_24db_image_dir = b_image_dir.split('.')[0] + '_24db.bin'
            qb_42db_image_dir = b_image_dir.split('.')[0] + '_42db.bin'
            b_image_file = os.path.join(B_folder, image_dir)
            qb_0db_image_file = os.path.join(QB_0db_folder, qb_0db_image_dir)
            qb_24db_image_file = os.path.join(QB_24db_folder, qb_24db_image_dir)
            qb_42db_image_file = os.path.join(QB_42db_folder, qb_42db_image_dir)
            self.b_files.append(b_image_file)
            self.qb_0db_files.append(qb_0db_image_file)
            self.qb_24db_files.append(qb_24db_image_file)
            self.qb_42db_files.append(qb_42db_image_file)
        self.nums_trainset = len(image_dir_list)
        
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        b_file = self.b_files[idx]
        qbs = [self.qb_0db_files[idx], self.qb_24db_files[idx], self.qb_24db_files[idx]]
        bayer, qbayer = b_file, qbs[random.randint(0, 2)]
        bayer, qbayer = read_bin_file(bayer), read_bin_file(qbayer) # h*w
        bayer, qbayer = bayer2rggb(bayer), qbayer2rggb(qbayer) # h//2 * w//2 * 4
        if self.train:
            bayer, qbayer = crop_patch(bayer, qbayer, self.patch_size, True)
            return bayer, qbayer
        return bayer, qbayer