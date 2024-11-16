# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from datasets.ProstateX_data import ProstateX_Dataset
from datasets.Prostate158_data import Prostate158_Dataset


def generate_dataset(dataset_name, cfg, period, aug=False):
	if dataset_name == 'ProstateX':
		return ProstateX_Dataset(cfg, period)
	if dataset_name == 'Prostate158':
		return Prostate158_Dataset(cfg, period)

	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
