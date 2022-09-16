import logging
import os
import struct
import re
import time

import numpy as np
import skimage.io as skio
import torch as th
# from scipy.stats import ortho_group
from skimage.color import rgb2hsv, hsv2rgb

from torch.utils.data import Dataset
from torchlib.image import read_pfm
from torchlib.utils import Timer

log = logging.getLogger("demosaic_data")

class SRGB2Linear(object):
    def __init__(self):
        super(SRGB2Linear, self).__init__()
        self.a = 0.055
        self.thresh = 0.04045
        self.scale = 12.92
        self.gamma = 2.4

    def __call__(self, im):
        return np.where(im <= self.thresh,
                        im / self.scale,
                        np.power((np.maximum(im, self.thresh) + self.a) / (1 + self.a), self.gamma))









class Linear2SRGB(object):
    def __init__(self):
        super(Linear2SRGB, self).__init__()
        self.a = 0.055
        self.thresh = 0.0031308049535603713
        self.scale = 12.92
        self.gamma = 2.4

    def __call__(self, im):
        return th.where(im <= self.thresh,
                        im * self.scale,
                        (1 + self.a)*th.pow(th.clamp(im, self.thresh), 1.0 / self.gamma) - self.a)







#[summary]
#   convert a RGB input image to bayer mosaic and mask
#[in]
#   im: input image. [3,m,n] ndarray
#
#[out]
#   bayer pattern of GR of the image. [3,m,n]
#                    BG
#
#   mask: mask of GR. [3,m,n]
#                 BG
#
def bayer_mosaic(im):
    """GRBG Bayer mosaic."""

    mos = np.copy(im)
    mask = np.ones_like(im)                 # create a ndarray of size 'im' filled with '1's

    # red
    mask[0, ::2, 0::2] = 0
    mask[0, 1::2, :] = 0

    # green
    mask[1, ::2, 1::2] = 0
    mask[1, 1::2, ::2] = 0

    # blue
    mask[2, 0::2, :] = 0
    mask[2, 1::2, 1::2] = 0

    return mos*mask, mask























###################################################################                 DemosaicDataset
#
#[summary]
#   DemosaicDataset defines the general operation for demosacing.
#   Depending on the CFA layout, a class will inherit from DemosaicDataset
#   to implement respective simulation from an input RGB image to mosaic
#
class DemosaicDataset(Dataset):


    # constructor of DemosaicDataset
    #[in]
    #   filelist: a txt file containing the list of image filenames used for training
    #   add_noise:
    #   max_noise:
    #   transform:
    #   augment:
    #   linearize:
    #
    def __init__(self, filelist, add_noise=False, max_noise=0.1, transform=None,
                 augment=False, linearize=False):


        self.transform = transform

        self.add_noise = add_noise
        self.max_noise = max_noise

        self.augment = augment

        if linearize:                                                                 #whether to train the model in linearize space or sRGB space
            self.linearizer = SRGB2Linear()
        else:
            self.linearizer = None

        if not os.path.splitext(filelist)[-1] == ".txt":
            raise ValueError("Dataset should be speficied as a .txt file")

        self.root = os.path.dirname(filelist)
        self.images = []


        with open(filelist) as fid:
            for l in fid.readlines():
                im = l.strip()
                self.images.append(os.path.join(self.root, im))
        self.count = len(self.images)



    def __len__(self):
        return self.count


    ################################
    #[summary]
    #
    #   this func is a virtual func to be implemnted by the class inheriting from the class
    #   this func is to make mosaic from an input RGB image depending on the CFA layout
    #
    def make_mosaic(self, im):
        return NotImplemented



    def __getitem__(self, idx):
        impath = self.images[idx]

        # read image
        im = skio.imread(impath).astype(np.float32) / 255.0

        # if self.augment:
        #   # Jitter the quantized values
        #   im += np.random.normal(0, 0.005, size=im.shape)
        #   im = np.clip(im, 0, 1)

        ########################################                                          data augmentation
        if self.augment:
            if np.random.uniform() < 0.5:
                im = np.fliplr(im)                                                                # flip the image left <--> right
            if np.random.uniform() < 0.5:
                im = np.flipud(im)                                                                # flip the image up <--> down

            im = np.rot90(im, k=np.random.randint(0, 4))

            # Pixel shift
            if np.random.uniform() < 0.5:
                shift_y = np.random.randint(0, 6)  # cover both xtrans and bayer
                im = np.roll(im, 1, 0)
            if np.random.uniform() < 0.5:
                shift_x = np.random.randint(0, 6)
                im = np.roll(im, 1, 1)

            # Random Hue/Sat
            if np.random.uniform() < 0.5:
                shift = np.random.uniform(-0.1, 0.1)
                sat = np.random.uniform(0.8, 1.2)
                im = rgb2hsv(im)
                im[:, :, 0] = np.mod(im[:, :, 0] + shift, 1)
                im[:, :, 1] *= sat
                im = hsv2rgb(im)

            im = np.clip(im, 0, 1)

        if self.linearizer is not None:
            im = self.linearizer(im)

        # Randomize exposure
        if self.augment:
            if np.random.uniform() < 0.5:
                im *= np.random.uniform(0.5, 1.2)

            im = np.clip(im, 0, 1)

            im = np.ascontiguousarray(im).astype(np.float32)

        im = np.transpose(im, [2, 1, 0])

        # crop boundaries to ignore shift
        c = 8
        im = im[:, c:-c, c:-c]

        ########################################                                          image augmentation
        #
        # get mosaic from an input image. This varies w/ different CFA layout
        mosaic, mask = self.make_mosaic(im)

        # TODO: separate GT/noisy
        # # add noise
        # std = 0
        # if self.add_noise:
        #   std = np.random.uniform(0, self.max_noise)
        #   im += np.random.normal(0, std, size=im.shape)
        #   im = np.clip(im, 0, 1)

        sample = {
            "mosaic": mosaic,                               # model input   [m,n,3]. unknown pixels are set to 0.
            "mask": mask,
            # "noise_variance": np.array([std]),
            "target": im,                                   # model output  [m,n,3]
        }

        # Augment
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        s = "Dataset\n"
        s += "  . {} images\n".format(len(self.images))
        return s


class ToBatch(object):
    def __call__(self, sample):
        for k in sample.keys():
            if type(sample[k]) == np.ndarray:
                sample[k] = np.expand_dims(sample[k], 0)
        return sample




# convert from ndarray to torch tensor
#
class ToTensor(object):
    def __call__(self, sample):
        for k in sample.keys():
            if type(sample[k]) == np.ndarray:
                sample[k] = th.from_numpy(sample[k])

        return sample

class GreenOnly(object):
    def __call__(self, sample):
        sample["target"][0] = 0
        sample["target"][2] = 0
        return sample






###################################################################                 Dataset for differnt CFAs


# Dataset for bayer CFA.
# inherits the DemosaicDataset and implements the virtual function: make_mosaic
class BayerDataset(DemosaicDataset):
    def make_mosaic(self, im):
        return bayer_mosaic(im)


class XtransDataset(DemosaicDataset):
    def make_mosaic(self, im):
        return xtrans_mosaic(im)
