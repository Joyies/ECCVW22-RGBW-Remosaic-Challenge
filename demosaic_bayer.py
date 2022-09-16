import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import demosaic.modules as modules
import demosaic.converter as converter
import pdb


def get_demosaic_net_model(pretrained, device, cfa='bayer', state_dict = False):
    '''
        get demosaic network
    :param pretrained:
        path to the demosaic-network model file [string]
    :param device:
        'cuda:0', e.g.
    :param state_dict:
        whether to use a packed state dictionary for model weights
    :return:
        model_ref: demosaic-net model

    '''

    model_ref = modules.get({"model": "BayerNetwork"})  # load model coefficients if 'pretrained'=True
    if not state_dict:
        cvt = converter.Converter(pretrained, "BayerNetwork")
        cvt.convert(model_ref)
        for p in model_ref.parameters():
            p.requires_grad = False
        model_ref = model_ref.to(device)
    else:
        model_ref.load_state_dict(torch.load(pretrained))
        model_ref = model_ref.to(device)

        model_ref.eval()

    return model_ref



def demosaic_by_demosaic_net(bayer, cfa, demosaic_net, device):
    '''
        demosaic the bayer to get RGB by demosaic-net. The func will covnert the numpy array to tensor for demosaic-net,
        after which the tensor will be converted back to numpy array to return.

    :param bayer:
        [m,n]. numpy float32 in the rnage of [0,1] linear bayer
    :param cfa:
        [string], 'RGGB', e.g. only GBRG, RGGB, BGGR or GRBG is supported so far!
    :param demosaic_net:
        demosaic_net object
    :param device:
        'cuda:0', e.g.

    :return:
        [m,n,3]. np array float32 in the rnage of [0,1]

    '''


    assert (cfa == 'GBRG') or (cfa == 'RGGB') or (cfa == 'GRBG') or (cfa == 'BGGR'), 'only GBRG, RGGB, BGGR, GRBG are supported so far!'

    # if the bayer resolution is too high (more than 1000x1000,e.g.), may cause memory error.

    bayer = np.clip(bayer ,0 ,1)
    bayer = torch.from_numpy(bayer).float()
    bayer = bayer.to(device)
    bayer = torch.unsqueeze(bayer, 0)
    bayer = torch.unsqueeze(bayer, 0)

    with torch.no_grad():

        rgb = predict_rgb_from_bayer_tensor(bayer, cfa=cfa, demosaic_net=demosaic_net, device=device)

    rgb = rgb.detach().cpu()[0].permute(1, 2, 0).numpy()  # torch tensor -> numpy array
    # rgb = np.clip(rgb, 0, 1)

    return rgb






def predict_rgb_from_bayer_tensor(im,cfa,demosaic_net,device):
    '''
        predict the RGB imgae from bayer pattern mosaic using demosaic net

    :param im:
        [batch_sz, 1, m,n] tensor. the bayer pattern mosiac.

    :param cfa:
        the cfa layout. the demosaic net is trained w/ GRBG. If the input is other than GRBG, need padding or cropping

    :param demosaic_net:
        demosaic-net

    :param device:
        'cuda:0', e.g.

    :return:
        rgb_hat:
          [batch_size, 3, m,n]  the rgb image predicted by the demosaic-net using our bayer input
    '''

    assert (cfa == 'GBRG') or (cfa == 'RGGB') or (cfa == 'GRBG') or (cfa == 'BGGR') , 'only GBRG, RGGB, BGGR, GRBG are supported so far!'

    # print(im.shape)

    n_channel = im.shape[1]

    if n_channel==1:            # gray scale image
        im= torch.cat((im, im, im), 1)


    if cfa == 'GBRG':       # the demosiac net is trained w/ GRBG
        im = pad_gbrg_2_grbg(im,device)
    elif cfa == 'RGGB':
        im = pad_rggb_2_grbg(im, device)
    elif cfa == 'BGGR':
        im = pad_bggr_2_grbg(im, device)



    im= bayer_mosaic_tensor(im,device)


    sample = {"mosaic": im}

    rgb_hat = demosaic_net(sample)

    if cfa == 'GBRG':
        # an extra row and col is padded on four sides of the bayer before using demosaic-net. Need to trim the padded rows and cols of demosaiced rgb
        rgb_hat = unpad_grbg_2_gbrg(rgb_hat)
    elif cfa == 'RGGB':
        rgb_hat = unpad_grbg_2_rggb(rgb_hat)
    elif cfa == 'BGGR':
        rgb_hat = unpad_grbg_2_bggr(rgb_hat)


    rgb_hat = torch.clamp(rgb_hat, min=0, max=1)

    return rgb_hat





def pad_bggr_2_grbg(bayer, device):
    '''
            pad bggr bayer pattern to get grbg (for demosaic-net)

        :param bayer:
            2d tensor [bsz,ch, h,w]
        :param device:
            'cuda:0' or 'cpu', or ...
        :return:
            bayer: 2d tensor [bsz,ch,h,w+2]

        '''
    bsz, ch, h, w = bayer.shape

    bayer2 = torch.zeros([bsz, ch, h + 2, w], dtype=torch.float32)
    bayer2 = bayer2.to(device)

    bayer2[:, :, 1:-1, :] = bayer

    bayer2[:, :,  0, :] = bayer[:, :, 1, :]
    bayer2[:, :, -1, :] = bayer2[:, :, -2, :]

    bayer = bayer2

    return bayer




def pad_rggb_2_grbg(bayer,device):
    '''
        pad rggb bayer pattern to get grbg (for demosaic-net)

    :param bayer:
        2d tensor [bsz,ch, h,w]
    :param device:
        'cuda:0' or 'cpu', or ...
    :return:
        bayer: 2d tensor [bsz,ch,h,w+2]

    '''
    bsz, ch, h, w = bayer.shape

    bayer2 = torch.zeros([bsz,ch,h, w+2], dtype=torch.float32)
    bayer2 = bayer2.to(device)

    bayer2[:,:,:, 1:-1] = bayer

    bayer2[:,:,:, 0] =  bayer[:,:,:, 1]
    bayer2[:,:,:, -1] = bayer2[:,:,:, -2]

    bayer = bayer2

    return bayer





def pad_gbrg_2_grbg(bayer,device):
    '''
        pad gbrg bayer pattern to get grbg (for demosaic-net)

    :param bayer:
        2d tensor [bsz,ch, h,w]
    :param device:
        'cuda:0' or 'cpu', or ...
    :return:
        bayer: 2d tensor [bsz,ch,h+4,w+4]

    '''
    bsz, ch, h, w = bayer.shape

    bayer2 = torch.zeros([bsz,ch,h+2, w+2], dtype=torch.float32)
    bayer2 = bayer2.to(device)

    bayer2[:,:,1:-1, 1:-1] = bayer
    bayer2[:,:,0, 1:-1] = bayer[:,:,1, :]
    bayer2[:,:,-1, 1:-1] = bayer[:,:,-2, :]

    bayer2[:,:,:, 0] =  bayer2[:,:,:, 2]
    bayer2[:,:,:, -1] = bayer2[:,:,:, -3]

    bayer = bayer2

    return bayer




def unpad_grbg_2_gbrg(rgb):
    '''
        unpad the rgb image. this is used after pad_gbrg_2_grbg()
    :param rgb:
        tensor. [1,3,m,n]
    :return:
        tensor [1,3,m-2,n-2]

    '''
    rgb = rgb[:,:,1:-1,1:-1]

    return rgb


def unpad_grbg_2_bggr(rgb):
    '''
           unpad the rgb image. this is used after pad_bggr_2_grbg()
       :param rgb:
           tensor. [1,3,m,n]
       :return:
           tensor [1,3,m,n-2]

       '''
    rgb = rgb[:, :, 1:-1 , : ]

    return rgb



def unpad_grbg_2_rggb(rgb):
    '''
        unpad the rgb image. this is used after pad_rggb_2_grbg()
    :param rgb:
        tensor. [1,3,m,n]
    :return:
        tensor [1,3,m,n-2]

    '''
    rgb = rgb[:,:,:,1:-1]

    return rgb



def bayer_mosaic_tensor(im,device):
    '''
        create bayer mosaic to set as input to demosaic-net.
        make sure the input bayer (im) is GRBG.

    :param im:
            [batch_size, 3, m,n]. The color is in RGB order.
    :param device:
            'cuda:0', e.g.
    :return:
    '''

    """GRBG Bayer mosaic."""

    batch_size=im.shape[0]
    hh=im.shape[2]
    ww=im.shape[3]

    mask = torch.ones([batch_size,3,hh, ww], dtype=torch.float32)
    mask = mask.to(device)

    # red
    mask[:,0, ::2, 0::2] = 0
    mask[:,0, 1::2, :] = 0

    # green
    mask[:,1, ::2, 1::2] = 0
    mask[:,1, 1::2, ::2] = 0

    # blue
    mask[:,2, 0::2, :] = 0
    mask[:,2, 1::2, 1::2] = 0

    return im*mask


