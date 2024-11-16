# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..', '..')) # /home/weixuakou2/
		self.EXP_NAME = 'reject'
		self.DATA_NAME = 'ProstateX'
		self.ORIENTATION = 'RAS'
		self.RandBiasFieldd = 1
		self.RandGaussianSmoothd = 1
		self.RandGibbsNoised = 1
		self.RandAffined =1
		self.RandRotate90d = 1
		self.RandRotated = 1
		self.RandElasticd = 1
		self.RandZoomd = 1
		self.RandCropByPosNegLabeld = 1
		self.RandGaussianNoised = 1
		self.RandShiftIntensityd = 1
		self.RandGaussianSharpend = 1
		self.RandAdjustContrastd = 1
		self.Resized = 256
		self.DATA_WORKERS = 4
		
		self.MODEL_NAME = 'rejector'
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'rejection/model/exp15')

		self.TRAIN_LR = 1e-5
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0.00001
		self.TRAIN_BN_MOM = 0.0003
		self.TRAIN_POWER = 0.9 
		self.TRAIN_GPUS = 1
		self.TRAIN_BATCHES =16
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0	
		self.TRAIN_EPOCHS = 400
		self.TRAIN_LOSS_LAMBDA = 0
		self.TRAIN_CKPT = None
		self.TEST_CKPT = None
		self.TEST_GPUS = 1
		self.TEST_BATCHES =16
		
		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'rejector/lib'))

		
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		#if self.TRAIN_GPUS != torch.cuda.device_count():
		#	raise ValueError('config.py: GPU number is not matched')
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)

cfg = Configuration() 	
