import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import datetime
import os
import sys
import cv2
from math import exp
import importlib
from pytorch_msssim import ssim
import struct

def mapping_mask(output, bayer_sg):

    b, c, h, w = output.shape
    mask_r2r = torch.zeros_like(output)
    mask_r2g = torch.zeros_like(output)
    mask_r2b = torch.zeros_like(output)
    mask_g2r = torch.zeros_like(output)
    mask_g2g = torch.zeros_like(output)
    mask_g2b = torch.zeros_like(output)
    mask_b2r = torch.zeros_like(output)
    mask_b2g = torch.zeros_like(output)
    mask_b2b = torch.zeros_like(output)

    mask_r2r[..., 0:h:4, 0:w:4] = 1
    mask_r2g[..., 0:h:4, 1:w:4] = 1
    mask_r2g[..., 1:h:4, 0:w:4] = 1
    mask_r2b[..., 1:h:4, 1:w:4] = 1

    mask_g2r[..., 0:h:4, 2:w:4] = 1
    mask_g2r[..., 2:h:4, 0:w:4] = 1
    mask_g2g[..., 0:h:4, 3:w:4] = 1
    mask_g2g[..., 1:h:4, 2:w:4] = 1
    mask_g2g[..., 2:h:4, 1:w:4] = 1
    mask_g2g[..., 3:h:4, 0:w:4] = 1
    mask_g2b[..., 1:h:4, 3:w:4] = 1
    mask_g2b[..., 3:h:4, 1:w:4] = 1

    mask_b2r[..., 2:h:4, 2:w:4] = 1
    mask_b2g[..., 2:h:4, 3:w:4] = 1
    mask_b2g[..., 3:h:4, 2:w:4] = 1
    mask_b2b[..., 3:h:4, 3:w:4] = 1

    loss_map = torch.abs(output - bayer_sg)
    loss = torch.mean(loss_map)
    loss_r2r = torch.sum(loss_map*mask_r2r) / torch.sum(mask_r2r)
    loss_r2g = torch.sum(loss_map*mask_r2g) / torch.sum(mask_r2g)
    loss_r2b = torch.sum(loss_map*mask_r2b) / torch.sum(mask_r2b)

    loss_g2r = torch.sum(loss_map*mask_g2r) / torch.sum(mask_g2r)
    loss_g2g = torch.sum(loss_map*mask_g2g) / torch.sum(mask_g2g)
    loss_g2b = torch.sum(loss_map*mask_g2b) / torch.sum(mask_g2b)

    loss_b2r = torch.sum(loss_map*mask_b2r) / torch.sum(mask_b2r)
    loss_b2g = torch.sum(loss_map*mask_b2g) / torch.sum(mask_b2g)
    loss_b2b = torch.sum(loss_map*mask_b2b) / torch.sum(mask_b2b)

    loss_set = {}
    loss_set['all'] = loss
    loss_set['r2r'] = loss_r2r
    loss_set['r2g'] = loss_r2g
    loss_set['r2b'] = loss_r2b
    loss_set['g2r'] = loss_g2r
    loss_set['g2g'] = loss_g2g
    loss_set['g2b'] = loss_g2b
    loss_set['b2r'] = loss_b2r
    loss_set['b2g'] = loss_b2g
    loss_set['b2b'] = loss_b2b

    return loss_set

    


def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 1.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)

def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, data_range=1, size_average=True)
    return float(ssim_val)

def save_bin(filepath, arr):
    '''
        save 2-d numpy array to '.bin' files with uint16

    @param filepath:
        expected file path to store data

    @param arr:
        2-d numpy array

    @return:
        None

    '''

    arr = np.round(arr).astype('uint16')
    arr = np.clip(arr, 0, 1023)
    height, width = arr.shape

    with open(filepath, 'wb') as fp:
        fp.write(struct.pack('<HH', width, height))
        arr.tofile(fp)

def read_bin_file(filepath):
    '''
        read '.bin' file to 2-d numpy array

    :param path_bin_file:
        path to '.bin' file

    :return:
        2-d image as numpy array (float32)

    '''

    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]

    data_2d = data[2:].reshape((hh, ww))
    data_2d = data_2d.astype(np.float32)

    return data_2d

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def import_module(name):
    return importlib.import_module(name)

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)

def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)
    
def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_stat_dict():
    stat_dict = {
        'epochs': 0,
        'losses': [],
        'ema_loss': 0.0,
        'scores_final': [],
        'scores_psnr': [],
        'scores_ssim': [],
        'scores_kld': [],
        'scores_lpips': [],
        'scores_niqe': [],
        'scores_musiq': [],
        'scores_final0': [],
        'scores_psnr0': [],
        'scores_ssim0': [],
        'scores_kld0': [],
        'scores_lpips0': [],
        'scores_niqe0': [],
        'scores_musiq0': [],
        'best_score_final': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_psnr': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_ssim': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_kld': {
            'value': 1.0,
            'epoch': 0
        },
        'best_score_lpips': {
            'value': 1.0,
            'epoch': 0
        },
        'best_score_niqe': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_musiq': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_final0': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_psnr0': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_ssim0': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_kld0': {
            'value': 1.0,
            'epoch': 0
        },
        'best_score_lpips0': {
            'value': 1.0,
            'epoch': 0
        },
        'best_score_niqe0': {
            'value': 0.0,
            'epoch': 0
        },
        'best_score_musiq0': {
            'value': 0.0,
            'epoch': 0
        }
        }
    return stat_dict

def prepare_qat(model):
    ## fuse model
    model.module.fuse_model()
    ## qconfig and qat-preparation & per-channel quantization
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    # model.qconfig = torch.quantization.QConfig(
    #     activation=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128,
    #         quant_max=127,
    #         qscheme=torch.per_tensor_symmetric,
    #         dtype=torch.qint8,
    #         reduce_range=False),
    #     weight=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128, 
    #         quant_max=+127, 
    #         dtype=torch.qint8, 
    #         qscheme=torch.per_tensor_symmetric, 
    #         reduce_range=False)
    # )
    model = torch.quantization.prepare_qat(model, inplace=True)
    return model
