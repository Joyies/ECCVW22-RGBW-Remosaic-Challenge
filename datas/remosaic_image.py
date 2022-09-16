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

def qbayer_sg2bayer_sg1(qbayer_sg):
    qbayer_rggb = qbayer2rggb(qbayer_sg)
    bayer_sg_bic = qbayer_rggb2bayer_sg(qbayer_rggb)

    return bayer_sg_bic

def qbayer_rggb2bayer_sg(qbayer_rggb):
    
    import cv2 as cv
    h, w, _ = qbayer_rggb.shape 
    qbayer_rggb_bic = cv.resize(qbayer_rggb, (w*2, h*2), interpolation=cv.INTER_CUBIC)
    rgb_bic = np.zeros((2*h, 2*w, 3))
    rgb_bic[..., 0] = qbayer_rggb_bic[..., 0]
    # rgb_bic[..., 1] = (qbayer_rggb_bic[..., 1] + qbayer_rggb_bic[..., 2]) / 2
    rgb_bic[..., 1] = qbayer_rggb_bic[..., 2]
    rgb_bic[..., 2] = qbayer_rggb_bic[..., 3]

    bayer_sg_bic = rgb2bayer(rgb_bic)
    
    return bayer_sg_bic

# R R G G R R G G R R G G
# R R G G R R G G R R G G
# G G B B G G B B G G B B
# G G B B G G B B G G B B
# R R G G R R G G R R G G
# R R G G R R G G R R G G
# G G B B G G B B G G B B
# G G B B G G B B G G B B
def qbayer_sg2bayer_sg(qbayer_sg):
    qbayer_sg = qbayer_sg.astype(np.float32)
    padding = 4
    qbayer_sg = np.pad(qbayer_sg, padding, 'reflect')
    bayer_sg = np.zeros_like(qbayer_sg).astype(np.float32)
    h, w = qbayer_sg.shape 
    # r2g, b2g
    # print(qbayer_sg[4, 6])
    # print()
    bayer_sg[4:h:4, 5:w:4] = (qbayer_sg[4:h:4, 6:w:4] + qbayer_sg[3:h-4:4, 5:w:4]) / 2. # (0, 1, g)
    bayer_sg[1:h:4, 4:w:4] = (qbayer_sg[2:h:4, 4:w:4] + qbayer_sg[1:h:4, 3:w-4:4]) / 2. # (1, 0, g)
    bayer_sg[2:h-4:4, 3:w-4:4] = (qbayer_sg[1:h-4:4, 3:w-4:4] + qbayer_sg[2:h-4:4, 4:w:4]) / 2. # (2, 3, g)
    bayer_sg[3:h-4:4, 2:w-4:4] = (qbayer_sg[3:h-4:4, 1:w-4:4] + qbayer_sg[4:h:4, 2:w-4:4]) / 2. # (3, 2, g)
    # g2r
    bayer_sg[4:h:4, 2:w:4] = qbayer_sg[4:h:4, 1:w:4]
    bayer_sg[2:h:4, 4:w:4] = qbayer_sg[1:h:4, 4:w:4]
    # g2b
    bayer_sg[1:h:4, 3:w:4] = qbayer_sg[2:h:4, 3:w:4]
    bayer_sg[3:h:4, 1:w:4] = qbayer_sg[3:h:4, 2:w:4]
    # r2b
    bayer_sg[2:h:4, 2:w:4] = qbayer_sg[1:h:4, 1:w:4]
    # b2r
    bayer_sg[1:h:4, 1:w:4] = qbayer_sg[2:h:4, 2:w:4]
    # r2r, grg, b2b
    bayer_sg[0:h:4, 0:w:4] = qbayer_sg[0:h:4, 0:w:4] # (0, 0, r)
    bayer_sg[0:h:4, 3:w:4] = qbayer_sg[0:h:4, 3:w:4] # (0, 3, g)
    bayer_sg[1:h:4, 2:w:4] = qbayer_sg[1:h:4, 2:w:4] # (1, 2, g)
    bayer_sg[2:h:4, 1:w:4] = qbayer_sg[2:h:4, 1:w:4] # (2, 1, g)
    bayer_sg[3:h:4, 0:w:4] = qbayer_sg[3:h:4, 0:w:4] # (3, 0, g)
    bayer_sg[3:h:4, 3:w:4] = qbayer_sg[3:h:4, 3:w:4] # (3, 3, b)

    bayer_sg = bayer_sg[padding:-padding, padding:-padding]
    
    return bayer_sg

# def qbayer_rggb2bayer_sg(qbayer_rggb):
    
#     import cv2 as cv
#     h, w, _ = qbayer_rggb.shape 
#     qbayer_rggb_bic = cv.resize(qbayer_rggb, (w*2, h*2), interpolation=cv.INTER_CUBIC)
#     qbayer_rggb_bic = np.clip(qbayer_rggb_bic, 0., 1023.)
#     rgb_q = np.zeros((h, w, 3))
#     rgb_q[..., 0] = qbayer_rggb[..., 0]
#     rgb_q[..., 1] = (qbayer_rggb[..., 1] + qbayer_rggb[..., 2]) / 2
#     rgb_q[..., 2] = qbayer_rggb[..., 3]
#     rgb_bic1 = np.zeros((2*h, 2*w, 3))
#     rgb_bic2 = np.zeros((2*h, 2*w, 3))
#     rgb_bic3 = np.zeros((2*h, 2*w, 3))
#     rgb_bic1[..., 0] = qbayer_rggb_bic[..., 0]
#     rgb_bic1[..., 1] = (qbayer_rggb_bic[..., 1] + qbayer_rggb_bic[..., 2]) / 2
#     rgb_bic1[..., 2] = qbayer_rggb_bic[..., 3]

#     rgb_bic2[..., 0] = qbayer_rggb_bic[..., 0]
#     rgb_bic2[..., 1] = qbayer_rggb_bic[..., 1]
#     rgb_bic2[..., 2] = qbayer_rggb_bic[..., 3]

#     rgb_bic3[..., 0] = qbayer_rggb_bic[..., 0]
#     rgb_bic3[..., 1] = qbayer_rggb_bic[..., 2]
#     rgb_bic3[..., 2] = qbayer_rggb_bic[..., 3]

#     plt.subplots(2, 2)
#     plt.subplot(2,2,1)
#     plt.imshow(rgb_bic1/1023)
#     plt.subplot(2,2,2)
#     plt.imshow(rgb_bic2/1023)
#     plt.subplot(2,2,3)
#     plt.imshow(rgb_bic3/1023)
#     plt.subplot(2,2,4)
#     plt.imshow(rgb_q/1023)
#     plt.savefig('g_channel.png')

#     bayer_sg_bic = rgb2bayer(rgb_bic)
    
#     return bayer_sg_bic

def rgb2bayer(rgb):
    h, w, _ = rgb.shape
    bayer = np.zeros((h, w))
    bayer[0:h:2, 0:w:2] = rgb[0:h:2, 0:w:2, 0]
    bayer[0:h:2, 1:w:2] = rgb[0:h:2, 1:w:2, 1]
    bayer[1:h:2, 0:w:2] = rgb[1:h:2, 0:w:2, 1] 
    bayer[1:h:2, 1:w:2] = rgb[1:h:2, 1:w:2, 2] 

    return bayer



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
    
def crop_patch(bayer_sg, qbayer_sg, qbayer_sg0, qbayer_sg24, qbayer_sg42, bayer, qbayer, qbayer0, qbayer24, qbayer42,patch_size, augment=False):
    # crop patch randomly
    h, w, _ = bayer.shape
    x, y = random.randrange(0, w - patch_size + 1, 2), random.randrange(0, h - patch_size + 1, 2)
    bayer, qbayer = bayer[y:y+patch_size, x:x+patch_size, :], qbayer[y:y+patch_size, x:x+patch_size, :]
    qbayer0, qbayer24, qbayer42 = qbayer0[y:y+patch_size, x:x+patch_size, :], qbayer24[y:y+patch_size, x:x+patch_size, :], qbayer42[y:y+patch_size, x:x+patch_size, :]
    bayer_sg, qbayer_sg = bayer_sg[2*y:2*(y+patch_size), 2*x:2*(x+patch_size)], qbayer_sg[2*y:2*(y+patch_size), 2*x:2*(x+patch_size)]
    qbayer_sg0, qbayer_sg24, qbayer_sg42 = qbayer_sg0[2*y:2*(y+patch_size), 2*x:2*(x+patch_size)], qbayer_sg24[2*y:2*(y+patch_size), 2*x:2*(x+patch_size)], qbayer_sg42[2*y:2*(y+patch_size), 2*x:2*(x+patch_size)]
    # bayer_sg_bic = qbayer_sg2bayer_sg(qbayer_sg)
    bayer_sg_bic = qbayer_sg2bayer_sg1(qbayer_sg)
    bayer_sg_bic = bayer_sg_bic[..., np.newaxis]
    bayer_sg = bayer_sg[..., np.newaxis]
    qbayer_sg, qbayer_sg0, qbayer_sg24, qbayer_sg42 = \
        qbayer_sg[..., np.newaxis], qbayer_sg0[..., np.newaxis], \
        qbayer_sg24[..., np.newaxis], qbayer_sg42[..., np.newaxis]

    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: 
            bayer_sg, qbayer_sg, bayer, qbayer = bayer_sg[:, ::-1, :], qbayer_sg[:, ::-1, :], bayer[:, ::-1, :], qbayer[:, ::-1, :]
            qbayer_sg0, qbayer_sg24, qbayer_sg42 = qbayer_sg0[:, ::-1, :], qbayer_sg24[:, ::-1, :], qbayer_sg42[:, ::-1, :]
            qbayer0, qbayer24, qbayer42 = qbayer0[:, ::-1, :], qbayer24[:, ::-1, :], qbayer42[:, ::-1, :]
            bayer_sg_bic = bayer_sg_bic[:, ::-1, :]
        if vflip: 
            bayer_sg, qbayer_sg, bayer, qbayer = bayer_sg[::-1, :, :], qbayer_sg[::-1, :, :], bayer[::-1, :, :], qbayer[::-1, :, :]
            qbayer_sg0, qbayer_sg24, qbayer_sg42 = qbayer_sg0[::-1, :, :], qbayer_sg24[::-1, :, :], qbayer_sg42[::-1, :, :]
            qbayer0, qbayer24, qbayer42 = qbayer0[::-1, :, :], qbayer24[::-1, :, :], qbayer42[::-1, :, :]
            bayer_sg_bic = bayer_sg_bic[::-1, :, :]
        if rot90: 
            bayer_sg, qbayer_sg, bayer, qbayer = bayer_sg.transpose(1,0,2), qbayer_sg.transpose(1,0,2), bayer.transpose(1,0,2), qbayer.transpose(1,0,2)
            qbayer_sg0, qbayer_sg24, qbayer_sg42 = qbayer_sg0.transpose(1,0,2), qbayer_sg24.transpose(1,0,2), qbayer_sg42.transpose(1,0,2)
            qbayer0, qbayer24, qbayer42 = qbayer0.transpose(1,0,2), qbayer24.transpose(1,0,2), qbayer42.transpose(1,0,2)
            bayer_sg_bic = bayer_sg_bic.transpose(1,0,2)
    # numpy to tensor
    bayer_sg_bic = ndarray2tensor(bayer_sg_bic)
    bayer_sg, qbayer_sg, bayer, qbayer = ndarray2tensor(bayer_sg), ndarray2tensor(qbayer_sg),  ndarray2tensor(bayer), ndarray2tensor(qbayer)
    qbayer_sg0, qbayer_sg24, qbayer_sg42 = ndarray2tensor(qbayer_sg0), ndarray2tensor(qbayer_sg24),  ndarray2tensor(qbayer_sg42)
    qbayer0, qbayer24, qbayer42 = ndarray2tensor(qbayer0), ndarray2tensor(qbayer24),  ndarray2tensor(qbayer42)
    return bayer_sg, qbayer_sg, qbayer_sg0, qbayer_sg24, qbayer_sg42, bayer, qbayer, qbayer0, qbayer24, qbayer42

class Remosaic(data.Dataset):
    def __init__(
        self, B_folder, QB_0db_folder, QB_24db_folder, QB_42db_folder, 
        train=True, augment=False, 
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

        if self.train:
            self.start_idx, self.end_idx = 1, 61
        else:
            self.start_idx, self.end_idx = 61, 71

        ## for raw files
        self.b_infoes = []
        self.b_files = []
        self.qb_0db_files = []
        self.qb_24db_files = []
        self.qb_42db_files = []
        ## for raw png images
        self.r_gains = []
        self.b_gains = []
        self.CCMs = []
        self.b_images = []
        self.qb_0db_images = []
        self.qb_24db_images = []
        self.qb_42db_images = []

        # len(image_dir_list)
        B_info_folder = B_folder.split('/GT_bayer')[0] + '/ImgInfo/train_RGBW_full_imgInfo'
        for i in range(self.start_idx, self.end_idx):
            idx = str(i).zfill(3)
            b_image_dir = 'rgbw_' + idx + '_fullres.bin'
            b_info_dir = b_image_dir.split('.')[0] + '.xml'
            qb_0db_image_dir = b_image_dir.split('.')[0] + '_0db.bin'
            qb_24db_image_dir = b_image_dir.split('.')[0] + '_24db.bin'
            qb_42db_image_dir = b_image_dir.split('.')[0] + '_42db.bin'
            b_info_file = os.path.join(B_info_folder, b_info_dir)
            b_image_file = os.path.join(B_folder, b_image_dir)
            qb_0db_image_file = os.path.join(QB_0db_folder, qb_0db_image_dir)
            qb_24db_image_file = os.path.join(QB_24db_folder, qb_24db_image_dir)
            qb_42db_image_file = os.path.join(QB_42db_folder, qb_42db_image_dir)
            self.b_infoes.append(b_info_file)
            self.b_files.append(b_image_file)
            self.qb_0db_files.append(qb_0db_image_file)
            self.qb_24db_files.append(qb_24db_image_file)
            self.qb_42db_files.append(qb_42db_image_file)
        
        LEN = self.end_idx - self.start_idx
        for i in range(LEN):
            if (i+1) % 5 == 0:
                print("read {} bin images!".format(i+1))  
            
            self.b_images.append(read_bin_file(self.b_files[i]))
            self.qb_0db_images.append(read_bin_file(self.qb_0db_files[i]))
            self.qb_24db_images.append(read_bin_file(self.qb_24db_files[i]))
            self.qb_42db_images.append(read_bin_file(self.qb_42db_files[i]))
        
        self.nums_trainset = LEN
        
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        info = self.b_infoes[idx]
        bayer = self.b_images[idx]
        
        if self.train:
            qbs = [self.qb_0db_images[idx], self.qb_24db_images[idx], self.qb_42db_images[idx]]
            qbayer0, qbayer24, qbayer42 = qbs[0], qbs[1], qbs[2]
            qbayer = qbs[random.randint(0, 2)]
            bayer_sg, qbayer_sg, qbayer_sg0, qbayer_sg24, qbayer_sg42 = bayer.copy(), qbayer.copy(),\
                qbayer0.copy(), qbayer24.copy(), qbayer42.copy()# h * w
            bayer_rggb, qbayer_rggb, qbayer_rggb0, qbayer_rggb24, qbayer_rggb42 = \
                bayer2rggb(bayer_sg), qbayer2rggb(qbayer_sg), qbayer2rggb(qbayer_sg0),\
                     qbayer2rggb(qbayer_sg24), qbayer2rggb(qbayer_sg42)# h//2 * w//2 * 4
            bayer_sg, qbayer_sg, qbayer_sg0, qbayer_sg24, qbayer_sg42, \
            bayer_rggb, qbayer_rggb, qbayer_rggb0, qbayer_rggb24, qbayer_rggb42 \
                = crop_patch(bayer_sg, qbayer_sg, qbayer_sg0, qbayer_sg24, qbayer_sg42, bayer_rggb, qbayer_rggb, qbayer_rggb0, qbayer_rggb24, qbayer_rggb42, self.patch_size, self.augment)
            back = {}
            back['bayer_sg'] = bayer_sg
            # back['bayer_sg_bic'] = bayer_sg_bic
            back['qbayer_sg'] = qbayer_sg
            back['qbayer_sg0'] = qbayer_sg0
            back['qbayer_sg24'] = qbayer_sg24
            back['qbayer_sg42'] = qbayer_sg42
            back['bayer_rggb'] = bayer_rggb
            back['qbayer_rggb0'] = qbayer_rggb0
            back['qbayer_rggb24'] = qbayer_rggb24
            back['qbayer_rggb42'] = qbayer_rggb42
            return back
        
        bayer_sg = bayer.copy() # h * w
        bayer_rggb = bayer2rggb(bayer_sg) # h//2 * w//2 * 4
        # get all qbayer_sg, h * w 
        qbayer_0db_sg = self.qb_0db_images[idx] 
        qbayer_24db_sg = self.qb_24db_images[idx]
        qbayer_42db_sg = self.qb_42db_images[idx]
        # get all qbayer_rggb, h//2 * w//2 * 4
        qbayer_0db_rggb = qbayer2rggb(qbayer_0db_sg)
        qbayer_24db_rggb = qbayer2rggb(qbayer_24db_sg)
        qbayer_42db_rggb = qbayer2rggb(qbayer_42db_sg)
        # get all bayer_xdb_bic, h * w 
        # bayer_0db_sg_bic = qbayer_sg2bayer_sg(qbayer_0db_sg)
        # bayer_24db_sg_bic = qbayer_sg2bayer_sg(qbayer_24db_sg)
        # bayer_42db_sg_bic = qbayer_sg2bayer_sg(qbayer_42db_sg)
        bayer_0db_sg_bic = qbayer_sg2bayer_sg1(qbayer_0db_sg)
        bayer_24db_sg_bic = qbayer_sg2bayer_sg1(qbayer_24db_sg)
        bayer_42db_sg_bic = qbayer_sg2bayer_sg1(qbayer_42db_sg)
        # get all qbayer_sg files
        bayer_sg_name = self.b_files[idx]
        qbayer_0db_sg_name = self.qb_0db_files[idx] 
        qbayer_24db_sg_name = self.qb_24db_files[idx]
        qbayer_42db_sg_name = self.qb_42db_files[idx]
        
    
        back = {}
        back['info'] = info
        back['bayer_sg'] = ndarray2tensor(bayer_sg[..., np.newaxis])
        back['bayer_0db_sg_bic'] = ndarray2tensor(bayer_0db_sg_bic[..., np.newaxis])
        back['bayer_24db_sg_bic'] = ndarray2tensor(bayer_24db_sg_bic[..., np.newaxis])
        back['bayer_42db_sg_bic'] = ndarray2tensor(bayer_42db_sg_bic[..., np.newaxis])
        back['qbayer_0db_sg'] = ndarray2tensor(qbayer_0db_sg[..., np.newaxis])
        back['qbayer_24db_sg'] = ndarray2tensor(qbayer_24db_sg[..., np.newaxis])
        back['qbayer_42db_sg'] = ndarray2tensor(qbayer_42db_sg[..., np.newaxis])
        back['bayer_rggb'] = ndarray2tensor(bayer_rggb)
        back['qbayer_0db_rggb'] = ndarray2tensor(qbayer_0db_rggb)
        back['qbayer_24db_rggb'] = ndarray2tensor(qbayer_24db_rggb)
        back['qbayer_42db_rggb'] = ndarray2tensor(qbayer_42db_rggb)
        back['bayer_sg_name'] = bayer_sg_name
        back['qbayer_0db_sg_name'] = qbayer_0db_sg_name
        back['qbayer_24db_sg_name'] = qbayer_24db_sg_name 
        back['qbayer_42db_sg_name'] = qbayer_42db_sg_name 

        return back

class Remosaic_val(data.Dataset):
    def __init__(
        self, QB_0db_folder, QB_24db_folder, QB_42db_folder, 
        train=True, test=False, augment=False, 
        patch_size=96, repeat=168
    ):
        super(Remosaic_val, self).__init__()
        self.augment   = augment
        self.img_postfix = '.bin'
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train
        self.test = test

        if self.train:
            self.start_idx, self.end_idx = 1, 61
        elif self.test:
            self.start_idx, self.end_idx = 71, 86 # test sequence
        else:
            self.start_idx, self.end_idx = 61, 71

        ## for raw files
        self.b_infoes = []
        self.b_files = []
        self.qb_0db_files = []
        self.qb_24db_files = []
        self.qb_42db_files = []
        ## for raw png images
        self.r_gains = []
        self.b_gains = []
        self.CCMs = []
        self.b_images = []
        self.qb_0db_images = []
        self.qb_24db_images = []
        self.qb_42db_images = []

        # len(image_dir_list)
        B_info_folder = QB_0db_folder.split('/input')[0] + '/ImgInfo/valid_RGBW_full_imgInfo'
        for i in range(self.start_idx, self.end_idx):
            idx = str(i).zfill(3)
            b_image_dir = 'rgbw_' + idx + '_fullres.bin'
            b_info_dir = b_image_dir.split('.')[0] + '.xml'
            qb_0db_image_dir = b_image_dir.split('.')[0] + '_0db.bin'
            qb_24db_image_dir = b_image_dir.split('.')[0] + '_24db.bin'
            qb_42db_image_dir = b_image_dir.split('.')[0] + '_42db.bin'
            b_info_file = os.path.join(B_info_folder, b_info_dir)
            # b_image_file = os.path.join(B_folder, b_image_dir)
            qb_0db_image_file = os.path.join(QB_0db_folder, qb_0db_image_dir)
            qb_24db_image_file = os.path.join(QB_24db_folder, qb_24db_image_dir)
            qb_42db_image_file = os.path.join(QB_42db_folder, qb_42db_image_dir)
            self.b_infoes.append(b_info_file)
            # self.b_files.append(b_image_file)
            self.qb_0db_files.append(qb_0db_image_file)
            self.qb_24db_files.append(qb_24db_image_file)
            self.qb_42db_files.append(qb_42db_image_file)
        
        LEN = self.end_idx - self.start_idx
        for i in range(LEN):
            if (i+1) % 5 == 0:
                print("read {} bin images!".format(i+1))  
            
            # self.b_images.append(read_bin_file(self.b_files[i]))
            self.qb_0db_images.append(read_bin_file(self.qb_0db_files[i]))
            self.qb_24db_images.append(read_bin_file(self.qb_24db_files[i]))
            self.qb_42db_images.append(read_bin_file(self.qb_42db_files[i]))
        
        self.nums_trainset = LEN
        
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        info = self.b_infoes[idx]
        # bayer = self.b_images[idx]
        qbs1 = self.qb_0db_images[idx]
        qbs2 = self.qb_24db_images[idx]
        qbs3 = self.qb_42db_images[idx]
        # qbayer = qbs[random.randint(0, 2)]
        # bayer_sg, qbayer_sg = bayer.copy(), qbayer.copy() # h * w
        qbayer_sg1 = qbs1.copy()
        qbayer_sg2 = qbs2.copy()
        qbayer_sg3 = qbs3.copy()

        # bayer_rggb, qbayer_rggb = bayer2rggb(bayer_sg), qbayer2rggb(qbayer_sg) # h//2 * w//2 * 4
        # qbayer_rggb1 = qbayer2rggb(qbayer_sg1)
        # qbayer_rggb2 = qbayer2rggb(qbayer_sg2)
        # qbayer_rggb3 = qbayer2rggb(qbayer_sg3)
        if self.train:
            bayer_sg, qbayer_sg, bayer_rggb, qbayer_rggb = crop_patch(bayer_sg, qbayer_sg, bayer_rggb, qbayer_rggb, self.patch_size, self.augment)
            back = {}
            back['bayer_sg'] = bayer_sg
            back['qbayer_sg'] = qbayer_sg
            back['bayer_rggb'] = bayer_rggb
            back['qbayer_rggb'] = qbayer_rggb
            return back
        
        
        back = {}
        back['info'] = info
        back['qbayer_0db_sg_name'] = self.qb_0db_files[idx]
        back['qbayer_24db_sg_name'] = self.qb_24db_files[idx]
        back['qbayer_42db_sg_name'] = self.qb_42db_files[idx]
        # back['bayer_sg'] = ndarray2tensor(bayer_sg[..., np.newaxis])
        back['qbayer_0db_sg'] = ndarray2tensor(qbayer_sg1[..., np.newaxis])
        back['qbayer_24db_sg'] = ndarray2tensor(qbayer_sg2[..., np.newaxis])
        back['qbayer_42db_sg'] = ndarray2tensor(qbayer_sg3[..., np.newaxis])
        # back['bayer_rggb'] = ndarray2tensor(bayer_rggb)
        # back['qbayer_rggb'] = ndarray2tensor(qbayer_rggb)

        return back

class Remosaic_test(data.Dataset):
    def __init__(
        self, QB_0db_folder, QB_24db_folder, QB_42db_folder, 
        train=True, test=False, augment=False, 
        patch_size=96, repeat=168
    ):
        super(Remosaic_test, self).__init__()
        self.augment   = augment
        self.img_postfix = '.bin'
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train
        self.test = test

        if self.train:
            self.start_idx, self.end_idx = 1, 61
        elif self.test:
            self.start_idx, self.end_idx = 86, 101#71, 86 # test sequence
        else:
            self.start_idx, self.end_idx = 61, 71

        ## for raw files
        self.b_infoes = []
        self.b_files = []
        self.qb_0db_files = []
        self.qb_24db_files = []
        self.qb_42db_files = []
        ## for raw png images
        self.r_gains = []
        self.b_gains = []
        self.CCMs = []
        self.b_images = []
        self.qb_0db_images = []
        self.qb_24db_images = []
        self.qb_42db_images = []

        # len(image_dir_list)
        B_info_folder = QB_0db_folder.split('/input')[0] + '/ImgInfo/test_RGBW_full_imgInfo'
        for i in range(self.start_idx, self.end_idx):
            idx = str(i).zfill(3)
            b_image_dir = 'rgbw_' + idx + '_fullres.bin'
            b_info_dir = b_image_dir.split('.')[0] + '.xml'
            qb_0db_image_dir = b_image_dir.split('.')[0] + '_0db.bin'
            qb_24db_image_dir = b_image_dir.split('.')[0] + '_24db.bin'
            qb_42db_image_dir = b_image_dir.split('.')[0] + '_42db.bin'
            b_info_file = os.path.join(B_info_folder, b_info_dir)
            # b_image_file = os.path.join(B_folder, b_image_dir)
            qb_0db_image_file = os.path.join(QB_0db_folder, qb_0db_image_dir)
            qb_24db_image_file = os.path.join(QB_24db_folder, qb_24db_image_dir)
            qb_42db_image_file = os.path.join(QB_42db_folder, qb_42db_image_dir)
            self.b_infoes.append(b_info_file)
            # self.b_files.append(b_image_file)
            self.qb_0db_files.append(qb_0db_image_file)
            self.qb_24db_files.append(qb_24db_image_file)
            self.qb_42db_files.append(qb_42db_image_file)
        
        LEN = self.end_idx - self.start_idx
        for i in range(LEN):
            if (i+1) % 5 == 0:
                print("read {} bin images!".format(i+1))  
            
            # self.b_images.append(read_bin_file(self.b_files[i]))
            self.qb_0db_images.append(read_bin_file(self.qb_0db_files[i]))
            self.qb_24db_images.append(read_bin_file(self.qb_24db_files[i]))
            self.qb_42db_images.append(read_bin_file(self.qb_42db_files[i]))
        
        self.nums_trainset = LEN
        
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        info = self.b_infoes[idx]
        # bayer = self.b_images[idx]
        qbs1 = self.qb_0db_images[idx]
        qbs2 = self.qb_24db_images[idx]
        qbs3 = self.qb_42db_images[idx]
        # qbayer = qbs[random.randint(0, 2)]
        # bayer_sg, qbayer_sg = bayer.copy(), qbayer.copy() # h * w
        qbayer_sg1 = qbs1.copy()
        qbayer_sg2 = qbs2.copy()
        qbayer_sg3 = qbs3.copy()

        # bayer_rggb, qbayer_rggb = bayer2rggb(bayer_sg), qbayer2rggb(qbayer_sg) # h//2 * w//2 * 4
        # qbayer_rggb1 = qbayer2rggb(qbayer_sg1)
        # qbayer_rggb2 = qbayer2rggb(qbayer_sg2)
        # qbayer_rggb3 = qbayer2rggb(qbayer_sg3)
        if self.train:
            bayer_sg, qbayer_sg, bayer_rggb, qbayer_rggb = crop_patch(bayer_sg, qbayer_sg, bayer_rggb, qbayer_rggb, self.patch_size, self.augment)
            back = {}
            back['bayer_sg'] = bayer_sg
            back['qbayer_sg'] = qbayer_sg
            back['bayer_rggb'] = bayer_rggb
            back['qbayer_rggb'] = qbayer_rggb
            return back
        
        
        back = {}
        back['info'] = info
        back['qbayer_0db_sg_name'] = self.qb_0db_files[idx]
        back['qbayer_24db_sg_name'] = self.qb_24db_files[idx]
        back['qbayer_42db_sg_name'] = self.qb_42db_files[idx]
        # back['bayer_sg'] = ndarray2tensor(bayer_sg[..., np.newaxis])
        back['qbayer_0db_sg'] = ndarray2tensor(qbayer_sg1[..., np.newaxis])
        back['qbayer_24db_sg'] = ndarray2tensor(qbayer_sg2[..., np.newaxis])
        back['qbayer_42db_sg'] = ndarray2tensor(qbayer_sg3[..., np.newaxis])
        # back['bayer_rggb'] = ndarray2tensor(bayer_rggb)
        # back['qbayer_rggb'] = ndarray2tensor(qbayer_rggb)

        return back