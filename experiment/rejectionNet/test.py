# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import torch.nn.functional as F
import cv2
# import surface_distance
from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from PIL import ImageDraw
import monai
from monai.data import DataLoader
from monai.metrics import DiceMetric
from scipy import ndimage


def train_net():
    test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
    test_dataloader = DataLoader(test_dataset, 
                batch_size=cfg.TEST_BATCHES, 
                shuffle=False, 
                num_workers=cfg.DATA_WORKERS)
    net = generate_net(cfg)
    print('Use %d GPU'%cfg.TRAIN_GPUS)
    device = torch.device(0)
    net.to(device)
    
    if cfg.TEST_CKPT:
        pretrained_dict = torch.load(cfg.TEST_CKPT_1)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
    net.eval()
    reject_ratio = {'total':0.0, 'reject':0.0}
    with torch.no_grad():
        test_hd = []
        test_dice = []
        test_score = []
        for i_batch, sample_batched in enumerate(test_dataloader):
            inputs_batched = sample_batched['image']
            labels_batched = sample_batched['segmentation']
            names_batched = sample_batched['name']
            h_batched, w_batched = sample_batched['h'], sample_batched['w']
            inputs_batched = inputs_batched.cuda()
            labels_batched = labels_batched.cuda()
            # predicts = net(x=inputs_batched)
            predicts, score_batched = net(x=inputs_batched, label=labels_batched, phase='test')
            predicts_batched = predicts.cuda()
            result_seg = torch.argmax(predicts_batched, dim=1, keepdim=True)
            # visualize_and_save(images=inputs_batched, preds=result_seg, labels=labels_batched,names=names_batched, save_path=r'/home/wx/rejection/plt_img/exp15')
            dice_target = cal_dice(result_seg, predicts_batched)
            test_dice.append(dice_target)
            test_score.append(score_batched)
            hd95 = cal_hau(pred=result_seg.cpu(), gt=labels_batched.cpu(), h_batch=h_batched, w_batch=w_batched)
            if hd95 is not None:
                test_hd.append(hd95)
        test_hd = np.concatenate(test_hd)
        test_dice = np.concatenate(test_dice)
        test_score = np.concatenate(test_score)
        print('test dice: %.2f + %.2f, HD: %.2f + %.2f'%(np.mean(test_dice), np.std(test_dice), np.mean(test_hd), np.std(test_hd)))
        # print('rejection ratio: %.2f'%(100.*reject_ratio['reject']/reject_ratio['total']))



def cal_R2(y, pred):
    mean_y = torch.mean(y)
    a = torch.sum((pred-y)**2)
    b = torch.sum((y-mean_y)**2)
    R2 = 1-a/b
    return R2     
    
def cal_dice(self, pred, y):
    intersection = (pred * y).sum(dim=(1, 2, 3))  
    pred_area = pred.sum(dim=(1, 2, 3)) 
    gt_area = y.sum(dim=(1, 2, 3)) 
    dice = (2 * intersection + 1e-6) / (pred_area + gt_area + 1e-6) 
    return dice

def cal_hau(pred, gt, h_batch, w_batch):
    b = pred.size(0)
    hd = []
    for i in range(b):
        pred_i = pred[i:i+1,:,:,:]
        gt_i = gt[i:i+1,:,:,:]
        height, weidth = h_batch[i], w_batch[i]
        current_h, current_w = pred_i.size(2), pred_i.size(3)
        # i_spacing = [float(0.4*height/current_h), float(0.4*weidth/current_w)]
        i_spacing = [float(0.5*height/current_h), float(0.5*weidth/current_w)]
        if torch.sum(pred_i)>0 and torch.sum(gt_i)>0:
            hau = monai.metrics.compute_hausdorff_distance(pred_i, gt_i, include_background=True, percentile=95, spacing=i_spacing)
            hd.append(hau.item())
    if len(hd)==0:
        hd_95 = None
    else:
        # hd_95 = np.max(hd)
        hd_95 = hd
    return hd_95
    



def get_params(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p



def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('%s has been saved'%new_file)






if __name__ == '__main__':
    train_net()


