import sys
import cv2
import numpy as np
from xml.etree import cElementTree as ET
import demosaic_bayer
from colour_demosaicing import (
    EXAMPLES_RESOURCES_DIRECTORY,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)
import os

class dmsc_method:
    demosaic_net = 0
    Menon2007 = 1

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


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

def ccm_corr(rgb, CCM, fwd=True):
    '''
        apply ccm color correction on linearized rgb

    :param rgb:
        [m,n,3]. numpy float32 array
    :param CCM:
        [3x3]. ccm matrix. numpy array.
    :param fwd:
        If True, output = CCM * rgb. If False, output = inv(CCM) * rgb

    :return:
        rgb: [m,n,3]. ccm corrected RGB.

    '''


    rgb = np.clip(rgb, 0, 1)

    h, w, c = rgb.shape

    assert c == 3, 'rgb need to be in shape of [h,w,3]'

    rgb = np.reshape(rgb, (h * w, c))

    if fwd:
        rgb_ccm = np.matmul(CCM, rgb.T)
    else:
        rgb_ccm = np.matmul(np.linalg.inv(CCM), rgb.T)

    rgb_ccm = np.reshape(rgb_ccm.T, (h, w, c))

    rgb_ccm = np.clip(rgb_ccm, 0, 1)

    return rgb_ccm

def read_simpleISP_imgIno(path):
    tree = ET.parse(path)
    root = tree.getroot()

    r_gain = root.find('r_gain').text
    b_gain = root.find('b_gain').text
    ccm_00 = root.find('ccm_00').text
    ccm_01 = root.find('ccm_01').text
    ccm_02 = root.find('ccm_02').text
    ccm_10 = root.find('ccm_10').text
    ccm_11 = root.find('ccm_11').text
    ccm_12 = root.find('ccm_12').text
    ccm_20 = root.find('ccm_20').text
    ccm_21 = root.find('ccm_21').text
    ccm_22 = root.find('ccm_22').text
    ccm_matrix = np.array([ccm_00, ccm_01, ccm_02,
                           ccm_10, ccm_11, ccm_12,
                           ccm_20, ccm_21, ccm_22])

    return float(r_gain), float(b_gain), ccm_matrix


def simple_ISP(bayer, cfa, r_gain, b_gain, CCM, dmsc = dmsc_method.demosaic_net):

    bayer = np.clip((bayer.astype(np.float32) - 64) / (1023 - 64), 0, 1)


    if dmsc == dmsc_method.demosaic_net:
        # Demosaic-net

        device = 'cuda:0'

        bayer = np.clip(np.power(bayer, 1 / 2.2), 0, 1)
        pretrained_model_path = os.path.dirname(__file__) + "/pretrained_models/bayer_p/model.bin"
        demosaic_net = demosaic_bayer.get_demosaic_net_model(pretrained=pretrained_model_path, device=device, cfa='bayer',
                                                             state_dict=True)
        rgb = demosaic_bayer.demosaic_by_demosaic_net(bayer=bayer, cfa=cfa, demosaic_net=demosaic_net, device=device)
        rgb = np.power(rgb, 2.2)
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb = demosaicing_CFA_Bayer_Menon2007(bayer, cfa)
        rgb = np.clip(rgb, 0, 1)



    # WBC
    rgb[:,:,0] *= r_gain
    rgb[:,:,2] *= b_gain

    rgb = np.clip(rgb, 0,1)

    # color correction
    CCM = np.asarray(CCM).astype(np.float32)
    CCM = CCM.reshape((3, 3))

    rgb = ccm_corr(rgb, CCM, fwd=True)  # (already clipped)

    # Gamma correction
    rgb = np.power(rgb, 1/2.2)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

def get_histogram(data, left_edge, right_edge, n_bins, bin_edges=None):

    '''
        Get normalized histogram from input data

    @param data:
        input data [numpy array]

    @param left_edge:
        x-axis left boundary of histogram [float num]

    @param right_edge:
        x-axis right boundary of histogram [float num]

    @param n_bins:
        number of expected bins [int]

    @param bin_edges:
        defined bin edges [None or 1-D numpy array]

    @return:
        - normalized histogram [1-D numpy array with size (n_bins,)]
        - bin edges used [1-D numpy array with size (n_bins + 1, )]
    '''

    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)

    #bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)



    return hist / n, bin_edges


def cal_kld_bayer(bayer_gt, bayer_hat):
    '''
        return symmetric KLD score from 10-bit bayer_gt (target) and bayer_hat (transformed)

    @param bayer_gt:
        input 2D image [numpy array] with range(0, 1023)

    @param bayer_hat:
        input 2D image [numpy array] with range(0, 1023)

    @return:
        symmetric KLD score [float num]
    '''

    # pdb.set_trace()
    # Get normalized histogram
    left_edge, right_edge, n_bins = 0, 1023, 1024
    h_gt, bin_edges = get_histogram(bayer_gt, left_edge, right_edge, n_bins, bin_edges=None)
    h_hat, bin_edges = get_histogram(bayer_hat, left_edge, right_edge, n_bins, bin_edges=None)


    # add one on zero elements
    ww, hh = bayer_gt.shape
    min_val = 1/(ww*hh)
    h_gt = np.where(h_gt != 0, h_gt, min_val)
    h_hat = np.where(h_hat != 0, h_hat, min_val)


    # KL_divergence: D_kl_fwd = sum{h(x) * log[h(x) / h_hat(x)]}
    kl_fwd = np.sum(h_gt  * (np.log(h_gt) - np.log(h_hat)))
    kl_inv = np.sum(h_hat * (np.log(h_hat) - np.log(h_gt)))

    return (kl_fwd + kl_inv)/2



def cal_kld_main(bayer_gt, bayer_out):
        '''
        calculate return symmetric KLD score from 10-bit bayer_gt (target) and bayer_hat (transformed)
        Each channel is calculated separately and its mean value is used.

        kld = (kld_gr + kld_r + kld_g + kld_gb)/4

        @param bayer_gt:
            input 2D image [numpy array] with range(0, 1023)

        @param bayer_hat:
            input 2D image [numpy array] with range(0, 1023)

        @return:
            symmetric KLD score [float num]
        '''
        score_channels = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                score_channels[i, j] = cal_kld_bayer(bayer_gt[i::2, j::2], bayer_out[i::2, j::2])

        # print(score_channels)
        score_mean = np.mean(score_channels)

        return score_mean