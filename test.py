import math
import argparse, yaml
import util
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import sys
sys.path.append("./") 
from datas.remosaic_image import rggb2bayer
import isp_util
from util import save_bin
import numpy as np

parser = argparse.ArgumentParser(description='EasySR')
parser.add_argument('--config', type=str, default='configs/stdrunet.yml', help = 'pre-config file for training')
parser.add_argument('--is_same_ensemble', type=str, default=True, help = 'pre-config file for training')
# parser.add_argument('--resume', type=str, default='experiments/scunet-one2one/', help = 'resume training or not')

def forward_x8(x, forward_function):
    def _transform(v, op):
        v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to('cuda')

        return ret

    lr_list = [x]
    for tf in 'v', 'h', 't':
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = [forward_function(aug) for aug in lr_list]
    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], 't')
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], 'h')
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], 'v')

    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)

    return output
    
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_test_datasets, create_val_datasets
    import matplotlib.pyplot as plt

    args = parser.parse_args()
    # the model path for test
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    
    test_model_path = args.test_model_path
    is_ensemble = True
    is_save_img = True
    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    
    ## definitions of model
    try:
        model = util.import_module('models.{}.{}_network'.format(args.model, args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    ## create dataset for test
    test_dataloader = create_test_datasets(args) # create_val_datasets

    # load test model
    print('load test model!')
    ckpt = torch.load(test_model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to('cuda')
    model = model.eval()

    ## create the lpips for calculating score
    import lpips
    loss_fn = lpips.LPIPS(net='alex', spatial = False) # 为了不重复 load 模型
    root = args.save_path
    root_bin = root + 'bin_ensamble_test/'
    root_img= root + 'img_ensamble_test/'
    if not os.path.exists(root_img):
        os.makedirs(root_img)
    if not os.path.exists(root_bin):
        os.makedirs(root_bin)

    for batch in tqdm(test_dataloader, ncols=80): 
        
        qbayer_0db_sg = batch['qbayer_0db_sg']
        qbayer_24db_sg = batch['qbayer_24db_sg']
        qbayer_42db_sg = batch['qbayer_42db_sg']
        qbayer_0db_sg_name = batch['qbayer_0db_sg_name']
        qbayer_24db_sg_name = batch['qbayer_24db_sg_name']
        qbayer_42db_sg_name = batch['qbayer_42db_sg_name']
        # bayer_sg_name = batch['bayer_sg_name']
        # bayer_rggb = batch['bayer_rggb'] 
        # qbayer_rggb = batch['qbayer_rggb']
        info = batch['info']
        r_gain, b_gain, CCM = isp_util.read_simpleISP_imgIno(info[0])
        qbayer_0db_sg = qbayer_0db_sg.to(device)
        qbayer_24db_sg = qbayer_24db_sg.to(device)
        qbayer_42db_sg = qbayer_42db_sg.to(device)
        qbs = [qbayer_0db_sg, qbayer_24db_sg, qbayer_42db_sg]
        qbs_name = [qbayer_0db_sg_name, qbayer_24db_sg_name, qbayer_42db_sg_name]
        
        for i in range(len(qbs)):
            qbayer_sg = qbs[i].to('cuda')
            qbayer_sg_name = qbs_name[i][0].split('/')[-1]
            if is_ensemble:
                output = forward_x8(qbayer_sg, model.forward)
            else:
                output = model(qbayer_sg)
            output = output.clamp(0, 1023)
            output = output.detach().squeeze(0).squeeze(0).cpu().numpy()
            if is_save_img:
                res_img = isp_util.simple_ISP(output, args.cfa, r_gain, b_gain, CCM, dmsc=args.dmsc)
                # save img
                res_img_path = root_img + qbs_name[i][0].split('/')[-1].split('.')[0] + '.png'
                isp_util.imwrite(res_img, res_img_path)
            # save bin
            bin_path = root_bin + qbayer_sg_name 
            save_bin(bin_path, output)


        
       
       
        

        