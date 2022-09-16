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
import pyiqa
import numpy as np
from util import mapping_mask
import lpips
                   
            

parser = argparse.ArgumentParser(description='EasySR')
## yaml configuration files
parser.add_argument('--config', type=str, default='configs/stdrunet.yml', help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets
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

    ## create dataset for training and validating
    train_dataloader, valid_dataloader = create_datasets(args)

    ## definitions of model
    try:
        model = util.import_module('models.{}.{}_network'.format(args.model, args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    ## definition of loss and optimizer
    lpips_metric = lpips.LPIPS(net='alex', spatial = False)
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(args.decays, args.gamma)
    if args.is_qat:
        scheduler = StepLR(optimizer, step_size=args.decays, gamma=args.gamma)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)

    ## load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['model_state_dict'])
        ## if qat
    if args.is_qat:
        if args.pretrain is not None:
            print('start quantization-awared training !')
            model = util.prepare_qat(model)
        else:
            raise ValueError('please provide pre-trained model for qat!')
    
    ## resume training
    start_epoch = 1
    if args.resume is not None:
        ckpt_files = glob.glob(os.path.join(args.resume, 'models', "*.pt"))
        if len(ckpt_files) != 0:
            ckpt_files = sorted(ckpt_files, key=lambda x: x.replace('.pt','').split('_')[-1])
            ckpt = torch.load(ckpt_files[-1])
            prev_epoch = ckpt['epoch']

            start_epoch = prev_epoch + 1
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            # print(scheduler)
            stat_dict = ckpt['stat_dict']
            ## reset folder and param
            experiment_path = args.resume
            log_name = os.path.join(experiment_path, 'log.txt')
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('select {}, resume training from epoch {}.'.format(ckpt_files[-1], start_epoch))
    else:
        ## auto-generate the output logname
        experiment_name = None
        timestamp = util.cur_timestamp_str()
        if args.log_name is None:
            experiment_name = '{}-{}'.format(args.model,timestamp)
        else:
            experiment_name = '{}-{}'.format(args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        log_name = os.path.join(experiment_path, 'log.txt')
        stat_dict = util.get_stat_dict()
        ## create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        ## save training paramters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    ## print architecture of model
    time.sleep(3) # sleep 3 seconds 
    sys.stdout = util.ExperimentLogger(log_name, sys.stdout)
    print(model)
    sys.stdout.flush()

    ## start training
    # for i in range(0, start_epoch):
    #     opt_lr = scheduler.get_last_lr()
    #     print('==>epoch:',i,'lr:',opt_lr)
    #     scheduler.step()
    timer_start = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print('##==========={}-training, Epoch: {}, lr: {} =============##'.format('int8' if args.is_qat else 'fp32', epoch, opt_lr))
        loss_all, loss_r2r, loss_r2g, loss_r2b, loss_g2r, loss_g2g, loss_g2b, loss_b2r, loss_b2g, loss_b2b = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            bayer_sg = batch['bayer_sg'] 
            qbayer_sg = batch['qbayer_sg'] 
            bayer_sg, qbayer_sg = bayer_sg.to(device), qbayer_sg.to(device)
            if args.data_range == 1:
                bayer_sg, qbayer_sg = bayer_sg/1023, qbayer_sg/1023
            output = model(qbayer_sg)
            loss_l1 = loss_func(output, bayer_sg)
            loss_set = mapping_mask(output, bayer_sg)
            if not args.mapping_loss:
                loss_l1.backward()
                optimizer.step()
            else:
                # loss_mapping = (loss_set['r2r'] + loss_set['r2g'] + loss_set['r2b'] + loss_set['b2r'] + \
                #     loss_set['b2g'] + loss_set['b2b'] + loss_set['g2r'] + loss_set['g2g'] + loss_set['g2b'])/9
                loss_all_tensor = (loss_set['r2r'] + loss_set['r2g'] + loss_set['r2b'] + loss_set['b2r'] + \
                     loss_set['b2g'] + loss_set['b2b'] + loss_set['g2r'] + loss_set['g2g'] + loss_set['g2b'])
                loss_mapping = (loss_all_tensor**2 - (loss_set['r2r']**2 + loss_set['r2g']**2 + loss_set['r2b']**2 + loss_set['b2r']**2 + loss_set['b2g']**2 + loss_set['b2b']**2 + loss_set['g2r']**2 + loss_set['g2g']**2 + loss_set['g2b']**2)) / (8 * loss_all_tensor)
                        
                loss_mapping.backward()
                optimizer.step()
            epoch_loss += float(loss_l1)
            # calulate different type mapping loss
            
            loss_all += float(loss_set['all'] )
            loss_r2r += float(loss_set['r2r'] )
            loss_r2g += float(loss_set['r2g'] )
            loss_r2b += float(loss_set['r2b'] )
            loss_g2r += float(loss_set['g2r'] )
            loss_g2g += float(loss_set['g2g'] )
            loss_g2b += float(loss_set['g2b'] )
            loss_b2r += float(loss_set['b2r'] )
            loss_b2g += float(loss_set['b2g'] )
            loss_b2b += float(loss_set['b2b'] )
            if (iter + 1) % args.log_every == 0:
                cur_steps = (iter+1)*args.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                stat_dict['losses'].append(avg_loss)

                avg_loss_all = loss_all / (iter + 1)
                avg_loss_r2r = loss_r2r / (iter + 1)
                avg_loss_r2g = loss_r2g / (iter + 1)
                avg_loss_r2b = loss_r2b / (iter + 1)
                avg_loss_g2r = loss_g2r / (iter + 1)
                avg_loss_g2g = loss_g2g / (iter + 1)
                avg_loss_g2b = loss_g2b / (iter + 1)
                avg_loss_b2r = loss_b2r / (iter + 1)
                avg_loss_b2g = loss_b2g / (iter + 1)
                avg_loss_b2b = loss_b2b / (iter + 1)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration))
                print('Mapping loss: R2R/R2B/R2G/G2R/G2G/G2B/B2R/B2G/B2B, {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'
                .format(avg_loss_r2r, avg_loss_r2g, avg_loss_r2b, avg_loss_g2r, avg_loss_g2g, avg_loss_g2b, avg_loss_b2r, avg_loss_b2g, avg_loss_b2b))
        
        if epoch % args.test_every == 0:
            # best practice for qat
            if epoch > 2 and args.is_qat:
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if epoch > 3 and args.is_qat:
                model.apply(torch.quantization.disable_observer)

            torch.set_grad_enabled(False)
            test_log = ''
            model = model.eval()
            metrics = ['psnr', 'ssim', 'lpips']
            nr_metrics = ['musiq', 'niqe']
            scores_final, scores_psnr, scores_ssim, scores_lpips, scores_kld, scores_niqe, scores_musiq = 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
            for batch in tqdm(valid_dataloader, ncols=80):
                gt = batch['bayer_sg'] 
                qbayer_0db_sg = batch['qbayer_0db_sg'] 
                qbayer_24db_sg = batch['qbayer_24db_sg'] 
                qbayer_42db_sg = batch['qbayer_42db_sg'] 
                info = batch['info']
                bayer_sg_name = batch['bayer_sg_name']
                qbayer_0db_sg_name = batch['qbayer_0db_sg_name'] 
                qbayer_24db_sg_name = batch['qbayer_24db_sg_name']
                qbayer_42db_sg_name  = batch['qbayer_42db_sg_name'] 
                r_gain, b_gain, CCM = isp_util.read_simpleISP_imgIno(info[0])
                qbs = [qbayer_0db_sg, qbayer_24db_sg, qbayer_42db_sg]
                qbs_name = [qbayer_0db_sg_name, qbayer_24db_sg_name, qbayer_42db_sg_name]
                for i in range(len(qbs)):
                    qbayer_sg = qbs[i]    
                    qbayer_sg = qbayer_sg.to(device)
                    if args.data_range == 1:
                        qbayer_sg = qbayer_sg/1023
                    output = model(qbayer_sg)
                    # quantize output to [0, 1023]
                    if args.data_range == 1:
                        output = output*1023
                    output = output.clamp(0, 1023)
                    # tensor to numpy
                    output = output.detach().squeeze(0).squeeze(0).cpu().numpy()
                    bayer_sg = gt.clone()
                    bayer_sg = bayer_sg.squeeze(0).squeeze(0).numpy()
                    
                    # calculate KLD
                    score_kld = isp_util.cal_kld_main(output, bayer_sg)
                    # bayer to RGB UINT8
                    res_img = isp_util.simple_ISP(output, args.cfa, r_gain, b_gain, CCM, dmsc=args.dmsc)
                    ref_img = isp_util.simple_ISP(bayer_sg, args.cfa, r_gain, b_gain, CCM, dmsc=args.dmsc)

                    # save RGB img
                    saved_img_dir = os.path.join(experiment_path, 'saved_img') 
                    if not os.path.exists(saved_img_dir):
                        os.makedirs(saved_img_dir)
                    if i == 0:
                        ref_img_path = saved_img_dir + '/' + bayer_sg_name[0].split('/')[-1].split('.')[0] + '.png'
                        isp_util.imwrite(ref_img, ref_img_path)
                    res_img_path = saved_img_dir + '/' + qbs_name[i][0].split('/')[-1].split('.')[0] + '.png'
                    isp_util.imwrite(res_img, res_img_path)
                    ref_img = ref_img.astype(np.float32)
                    res_img = res_img.astype(np.float32)
                    # evaluate with 3 metrics
                    res_img = torch.tensor(res_img).permute(2, 0, 1).unsqueeze_(0) / 255.
                    ref_img = torch.tensor(ref_img).permute(2, 0, 1).unsqueeze_(0) / 255.
                    # Reference assessment
                    iqa_metric = pyiqa.create_metric('psnr')
                    score_psnr = iqa_metric(res_img, ref_img).item()
                    iqa_metric = pyiqa.create_metric('ssim')
                    score_ssim = iqa_metric(res_img, ref_img).item()
                    
                    score_lpips = lpips_metric.forward(res_img, ref_img).item()
                    # iqa_metric = pyiqa.create_metric('lpips')
                    # score_lpips = iqa_metric(res_img, ref_img).item()
                    # Non-Reference assessment
                    score_musiq = 1
                    # iqa_metric = pyiqa.create_metric('musiq')
                    # score_musiq = iqa_metric(res_img).item()  
                    score_niqe = 1
                    # iqa_metric = pyiqa.create_metric('niqe')
                    # score_niqe = iqa_metric(res_img).item()  
                    # add to scores
                    scores_psnr += score_psnr
                    scores_ssim += score_ssim
                    scores_lpips += score_lpips
                    scores_kld += score_kld
                    scores_musiq += score_musiq
                    scores_niqe += score_niqe
                    scores_final +=  score_psnr * score_ssim  * 2 ** (1 - score_kld - score_lpips)
                
            avg_score_final = round(scores_final / (3 * (len(valid_dataloader))) + 5e-3, 4)
            avg_score_psnr = round(scores_psnr / (3 * (len(valid_dataloader))) + 5e-3, 4)
            avg_score_ssim = round(scores_ssim / (3 * (len(valid_dataloader))) + 5e-3, 4)
            avg_score_lpips = round(scores_lpips / (3 * (len(valid_dataloader))) + 5e-3, 4)
            avg_score_kld = round(scores_kld / (3 * (len(valid_dataloader))) + 5e-3, 4)
            avg_score_musiq = round(scores_musiq / (3 * (len(valid_dataloader))) + 5e-3, 4)
            avg_score_niqe = round(scores_niqe / (3 * (len(valid_dataloader))) + 5e-3, 4)

            stat_dict['scores_final'].append(avg_score_final)
            stat_dict['scores_psnr'].append(avg_score_psnr)
            stat_dict['scores_ssim'].append(avg_score_ssim)
            stat_dict['scores_kld'].append(avg_score_kld)
            stat_dict['scores_lpips'].append(avg_score_lpips)
            
            if stat_dict['best_score_final']['value'] < avg_score_final:
                stat_dict['best_score_final']['value'] = avg_score_final
                stat_dict['best_score_final']['epoch'] = epoch
            if stat_dict['best_score_psnr']['value'] < avg_score_psnr:
                stat_dict['best_score_psnr']['value'] = avg_score_psnr
                stat_dict['best_score_psnr']['epoch'] = epoch
            if stat_dict['best_score_ssim']['value'] < avg_score_ssim:
                stat_dict['best_score_ssim']['value'] = avg_score_ssim
                stat_dict['best_score_ssim']['epoch'] = epoch
            if stat_dict['best_score_kld']['value'] > avg_score_kld:
                stat_dict['best_score_kld']['value'] = avg_score_kld
                stat_dict['best_score_kld']['epoch'] = epoch
            if stat_dict['best_score_lpips']['value'] > avg_score_lpips:
                stat_dict['best_score_lpips']['value'] = avg_score_lpips
                stat_dict['best_score_lpips']['epoch'] = epoch
            test_log += 'Score/PSNR/SSIM/LPIPS/KLD: {:.4f}/{:.2f}/{:.4f}/{:.4f}/{:.4f}, \
                (Best Score: {:.4f}/{:.2f}/{:.4f}/{:.4f}/{:.4f}, Epoch: {}/{}/{}/{}/{} ) \n'.format(
                    float(avg_score_final), float(avg_score_psnr), float(avg_score_ssim), float(avg_score_lpips), float(avg_score_kld),
                    stat_dict['best_score_final']['value'], stat_dict['best_score_psnr']['value'], stat_dict['best_score_ssim']['value'],
                    stat_dict['best_score_lpips']['value'], stat_dict['best_score_kld']['value'],
                    stat_dict['best_score_final']['epoch'], stat_dict['best_score_psnr']['epoch'], stat_dict['best_score_ssim']['epoch'],
                    stat_dict['best_score_lpips']['epoch'], stat_dict['best_score_kld']['epoch'])
            # print log & flush out
            print(test_log)
            sys.stdout.flush()
            # save model
            saved_model_path = os.path.join(experiment_model_path, 'model_latest.pt')
            # torch.save(model.state_dict(), saved_model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stat_dict': stat_dict
            }, saved_model_path)
            if epoch == stat_dict['best_score_final']['epoch']:
                saved_model_path = os.path.join(experiment_model_path, 'model_best.pt')
                # torch.save(model.state_dict(), saved_model_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'stat_dict': stat_dict
                }, saved_model_path)
            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        ## update scheduler
        scheduler.step()