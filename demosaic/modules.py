import os
import sys

import copy
import time
from collections import OrderedDict

import numpy as np
from torch.autograd import Variable
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchlib.modules import Autoencoder
from torchlib.modules import ConvChain
from torchlib.image import crop_like
# import torchlib.viz as viz

from demosaic.dataset import Linear2SRGB

# import torchlib.debug as D


def apply_kernels(kernel, noisy_data):
    kh, kw = kernel.shape[2:]
    bs, ci, h, w = noisy_data.shape
    ksize = int(np.sqrt(kernel.shape[1]))

    # Crop kernel and input so their sizes match
    needed = kh + ksize - 1
    if needed > h:
        crop = (needed - h) // 2
        if crop > 0:
            kernel = kernel[:, :, crop:-crop, crop:-crop]
        kh, kw = kernel.shape[2:]
    else:
        crop = (h - needed) // 2
        if crop > 0:
            noisy_data = noisy_data[:, :, crop:-crop, crop:-crop]

    # -------------------------------------------------------------------------
    # Vectorize the kernel tiles
    kernel = kernel.permute(0, 2, 3, 1)
    kernel = kernel.contiguous().view(bs, 1, kh, kw, ksize*ksize)

    # Split the input buffer in tiles matching the kernels
    tiles = noisy_data.unfold(2, ksize, 1).unfold(3, ksize, 1)
    tiles = tiles.contiguous().view(bs, ci, kh, kw, ksize*ksize)
    # -------------------------------------------------------------------------

    weighted_sum = th.sum(kernel*tiles, dim=4)

    return weighted_sum









########################################################                      load the model structrue based on params                    ########################################################

def get(params):
    params = copy.deepcopy(params)  # do not touch the original

    # get the model name from the input params
    model_name = params.pop("model", None)


    if model_name is None:
        raise ValueError("model has not been specified!")

    # get the model structure by model_name
    return getattr(sys.modules[__name__], model_name)(**params)




#####################################################33                                             BayerNetwork



class BayerNetwork(nn.Module):
    """Released version of the network, best quality.

    This model differs from the published description. It has a mask/filter split
    towards the end of the processing. Masks and filters are multiplied with each
    other. This is not key to performance and can be ignored when training new
    models from scratch.
    """
    def __init__(self, depth=15, width=64):
        super(BayerNetwork, self).__init__()

        self.depth = depth
        self.width = width

        # self.debug_layer = nn.Conv2d(3, 4, 2, stride=2)
        # self.debug_layer1 =nn.Conv2d(in_channels=4,out_channels=64,kernel_size=3,stride=1,padding=1)
        # self.debug_layer2 =nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        # self.debug_layer3 =nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)

        layers = OrderedDict([
            ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance.
        ])                                                  #
                                                            # the output of 'pack_mosaic' will be half width and height of the input
                                                            # [batch_size, 4, h/2, w/2] = pack_mosaic ( [batch_size, 3, h, w] )

        for i in range(depth):
            #num of in and out neurons in each layers
            n_out = width
            n_in = width

            if i == 0:                          # the 1st layer in main_processor
                n_in = 4
            if i == depth-1:                    # the last layer in main_processor
                n_out = 2*width


            # layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
            layers["conv{}".format(i + 1)] = nn.Conv2d(n_in, n_out, 3,stride=1,padding=1)           # padding is set to be 1 so that the h and w won't change after conv2d (using kernal size 3)

            layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)


        # main conv layer
        self.main_processor = nn.Sequential(layers)

        # residual layer
        self.residual_predictor = nn.Conv2d(width, 12, 1)

        # upsample layer
        self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

        # full-res layer
        self.fullres_processor = nn.Sequential(OrderedDict([
            # ("post_conv", nn.Conv2d(6, width, 3)),
            ("post_conv", nn.Conv2d(6, width, 3,stride=1,padding=1)),                                # padding is set to be 1 so that the h and w won't change after conv2d (using kernal size 3)
            ("post_relu", nn.ReLU(inplace=True)),
            ("output", nn.Conv2d(width, 3, 1)),
        ]))




    # samples structure
    #   sample = {
    #         "mosaic": mosaic,                               # model input   [batch_size, 3, h,w]. unknown pixels are set to 0.
    #         "mask": mask,
    #         # "noise_variance": np.array([std]),
    #         "target": im,                                   # model output  [m,n,3]
    #     }
    def forward(self, samples):

        mosaic = samples["mosaic"]                                                  # [batch_size, 3, h, w]

        features = self.main_processor(mosaic)                                      # [batch_size, self.width*2, hf,wf]

        filters, masks = features[:, :self.width], features[:, self.width:]

        filtered = filters * masks                                                  # [batch_size, self.width, hf,wf]

        residual = self.residual_predictor(filtered)                                # [batch_size, 12, hf, wf]

        upsampled = self.upsampler(residual)                                        # [batch_size, 3, hf*2, wf*2]. upsampled will be 2x2 upsample of residual using ConvTranspose2d()

        # crop original mosaic to match output size
        cropped = crop_like(mosaic, upsampled)

        # Concated input samples and residual for further filtering
        packed = th.cat([cropped, upsampled], 1)

        output = self.fullres_processor(packed)

        return output







class XtransNetwork(nn.Module):
  """Released version of the network.
  There is no downsampling here.
  """
  def __init__(self, depth=11, width=64):
    super(XtransNetwork, self).__init__()

    self.depth = depth
    self.width = width

    layers = OrderedDict([])
    for i in range(depth):
      n_in = width
      n_out = width
      if i == 0:
        n_in = 3
      # if i == depth-1:
      #   n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(3+width, width, 3)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

  def forward(self, samples):
    # 1/4 resolution features
    mosaic = samples["mosaic"]
    features = self.main_processor(mosaic)

    # crop original mosaic to match output size
    cropped = crop_like(mosaic, features)

    # Concated input samples and residual for further filtering
    packed = th.cat([cropped, features], 1)

    output = self.fullres_processor(packed)

    return output


