from datasets.transform import *
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import monai
import pandas as pd
from monai.data import CacheDataset
def Prostate158_Dataset(cfg, period):
    root_dir = r'/home/wx/rejection/data'
    train_f = pd.read_csv(os.path.join(root_dir, 'ImageSets00/%s.txt'%period))
    train_ids = train_f.iloc[:,0]
    data_dict = []
    for i in range(len(train_ids)):
        img_path = os.path.join(root_dir, 'Task00_t2w', train_ids[i])
        adc_path = os.path.join(root_dir, 'Task00_adc', train_ids[i])
        dwi_path = os.path.join(root_dir, 'Task00_dwi', train_ids[i])
        label_path = os.path.join(root_dir, 'Task00_mask', train_ids[i])
        img_lists = os.listdir(img_path)
        for j in range(len(img_lists)):
            img_id = int(img_lists[j].split('.')[0])
            img_name = '%s_%d'%(train_ids[i], img_id)
            img_path_j = os.path.join(img_path, img_lists[j])
            adc_path_j = os.path.join(adc_path, img_lists[j])
            dwi_path_j = os.path.join(dwi_path, img_lists[j])
            label_path_j = os.path.join(label_path, img_lists[j])
            data_dict.append({'t2w':img_path_j, 'adc':adc_path_j, 'dwi':dwi_path_j, 'segmentation':label_path_j, 'name':img_name})
    if period =='train':
        transforms = get_train_transforms(cfg)
    elif period =='val':
        transforms = get_val_transforms()
    elif period =='test':
        transforms = get_test_transforms()
    dataset = CacheDataset(data=data_dict, transform=transforms)
    return dataset








