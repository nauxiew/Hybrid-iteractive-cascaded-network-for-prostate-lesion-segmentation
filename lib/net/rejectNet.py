import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import scipy.ndimage
from monai.networks.layers.factories import Act, Norm
from net.Unet2D import Backbone
from skimage import measure
from monai.networks.nets import EfficientNetBN

class EfficientNetRegressor(nn.Module):
    def __init__(self, in_c):
        super(EfficientNetRegressor, self).__init__()
        self.in_c = in_c
        self.weight_path = r'/home/wx/rejection/model/efficientnet-b5-b6417697.pth'
        self.efficient_net = EfficientNetBN(model_name="efficientnet-b5", pretrained=False, spatial_dims=2, in_channels=self.in_c, num_classes=512)
        net_dict = self.efficient_net.state_dict()
        state_dict = torch.load(self.weight_path)
        state_dict = {k:v for k,v in state_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(state_dict)
        self.efficient_net.load_state_dict(net_dict)
        self.fc1 = nn.Linear(512, 256) 
        self.fc2 = nn.Linear(256, 1)    
        self.relu = nn.ReLU()           
    def forward(self, x):
        x = self.efficient_net(x)       
        x = self.relu(self.fc1(x))       
        x = self.relu(self.fc2(x))
        return x


class RejectNet(nn.Module):
    def __init__(self):
        super(RejectNet, self).__init__()        
        self.encoder1 = Backbone(in_c = 3)
        self.rejector = EfficientNetRegressor(in_c=32+2)
        
    def cal_error_loss(self, score, pred, y):
        pred = torch.argmax(pred, dim=1, keepdim=True)
        intersection = (pred * y).sum(dim=(1, 2, 3))  
        pred_area = pred.sum(dim=(1, 2, 3)) 
        gt_area = y.sum(dim=(1, 2, 3)) 
        dice = (2 * intersection + 1e-6) / (pred_area + gt_area + 1e-6) 
        loss = F.mse_loss(score[:,0], dice).cuda()
        return loss.cuda()

        
    def forward(self, x, label=None, phase='train'):
        ### for coarse segmentaiton train
        out_1,_ = self.encoder1(x)
        return out_1
        ###### for coarse + rejector
        # out_1, f_deep = self.encoder1(x)
        # p_1 = F.softmax(out_1, dim=1)
        # x_in = torch.cat((f_deep.detach(), p_1.detach()), dim=1)
        # score = self.rejector(x_in)
        # mse_loss = self.cal_error_loss(score=score, pred=out_1, y=label)
        # if phase=='train':
            # return out_1, mse_loss, score
        # else:
            # return out_1, score
       

    




