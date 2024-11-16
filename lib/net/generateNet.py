# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from net.rejectNet import RejectNet

def generate_net(cfg):
	if cfg.MODEL_NAME == 'rejector':
		return RejectNet()
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)

