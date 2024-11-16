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
import monai
from monai.data import DataLoader
from monai.metrics import DiceMetric


def train_net():
    # laod segmentation data
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train')
    dataloader = DataLoader(dataset, 
                batch_size=cfg.TRAIN_BATCHES, 
                shuffle=cfg.TRAIN_SHUFFLE, 
                num_workers=cfg.DATA_WORKERS,
                drop_last=True) 
    val_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'val')
    val_dataloader = DataLoader(val_dataset, 
                batch_size=cfg.TEST_BATCHES, 
                shuffle=False, 
                num_workers=cfg.DATA_WORKERS)
    net = generate_net(cfg)
    print('Use %d GPU'%cfg.TRAIN_GPUS)
    device = torch.device(0)
    net.to(device)
    
    if cfg.TRAIN_CKPT:
        pretrained_dict = torch.load(cfg.TRAIN_CKPT)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
    criterion = nn.CrossEntropyLoss()
    dice_criterion = monai.losses.DiceLoss(softmax=True, to_onehot_y=True, smooth_nr=1e-6, smooth_dr =1e-6)

    optimizer = optim.Adam(params = [
            {'params': get_params(net,key='encoder1'), 'lr': cfg.TRAIN_LR},
            {'params': get_params(net,key='other'), 'lr': 10*cfg.TRAIN_LR},
        ])
    for para in optimizer.param_groups[0]['params']:
        para.requires_grad=False
    best_jacc = 0.
    best_epoch = 0
    # best_jacc = 100.
    for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
        loss_batched = 0.0
        net.train()
        for i_batch, sample_batched in enumerate(dataloader):
            inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
            h_batched, w_batched = sample_batched['h'], sample_batched['w']
            inputs_batched = inputs_batched.cuda()
            labels_batched = labels_batched.cuda().long()
            optimizer.zero_grad()
            predicts_batched, mse_loss, score_target = net(x=inputs_batched, label=labels_batched, phase='train')
            # predicts_batched = net(x=inputs_batched, label=labels_batched)

            predicts_batched = predicts_batched.cuda()         
            predicts_batched_binared = torch.argmax(predicts_batched, dim=1, keepdim=True)  
            dice_loss = dice_criterion(predicts_batched, labels_batched)
            ce_loss = criterion(predicts_batched, labels_batched[:,0,:,:])
            total_loss = (dice_loss + ce_loss + mse_loss)
            total_loss.backward()
            optimizer.step()
            loss_batched += total_loss.item()        
        print('epoch:%d/%d\t  loss:%g\t \n' %(epoch, cfg.TRAIN_EPOCHS, loss_batched/i_batch))        
        net.eval()
        with torch.no_grad():
            test_hd = []
            test_dice = []
            test_score = []
            for i_batch, sample_batched in enumerate(val_dataloader):
                inputs_batched = sample_batched['image']
                labels_batched = sample_batched['segmentation']
                h_batched, w_batched = sample_batched['h'], sample_batched['w']
                inputs_batched = inputs_batched.cuda()
                labels_batched = labels_batched.cuda()
                predicts, score_batched = net(x=inputs_batched, label=labels_batched, phase='test')
                # predicts = net(x=inputs_batched, label=labels_batched)
                predicts_batched = predicts.cuda()
                result_seg = torch.argmax(predicts_batched, dim=1, keepdim=True)
                dice_target = cal_dice(result_seg, predicts_batched)
                test_dice.append(dice_target)
                test_score.append(score_batched)
                hd95 = cal_hau(pred=result_seg.cpu(), gt=labels_batched.cpu(), h_batch=h_batched, w_batch=w_batched)
                if hd95 is not None:
                    test_hd.append(hd95)
            if len(test_hd)>0:
                test_hd = np.concatenate(test_hd)
            test_dice = np.concatenate(test_dice)
            test_score = np.concatenate(test_score)
            r2 = cal_R2(test_dice, test_score)
            print('test dice: %.2f + %.2f, HD: %.2f + %.2f'%(np.mean(test_dice), np.std(test_dice), np.mean(test_hd), np.std(test_hd)))
            print('r2: %.2f'%r2)
            if r2>best_jacc:
                model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.DATA_NAME,epoch,r2)),
                        old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.DATA_NAME,best_epoch,best_jacc)))
                best_jacc = r2
                best_epoch = epoch


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
        if key == 'encoder1':
            if 'encoder1' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == 'other':
            if 'encoder1' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p




def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('%s has been saved'%new_file)



if __name__ == '__main__':
    train_net()


