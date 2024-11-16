import os
import numpy as np
import torch
import scipy.ndimage
from monai.utils.enums import CommonKeys
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    SaveImaged,
    ScaleIntensityd,
    NormalizeIntensityd,
    Resized,
    AddExtremePointsChanneld,
)
from monai.transforms import Resized
import GeodisTK


class CatImage():
    def __init__(self, keys):
        self.t2w_key, self.adc_key, self.dwi_key = keys
    def __call__(self, data):
        d = dict(data)
        t2w =d[self.t2w_key]
        adc = d[self.adc_key]
        dwi = d[self.dwi_key]
        img = torch.cat((t2w,adc, dwi),dim=0)
        d['image'] = img
        return d
class SaveDim():
    def __init__(self, keys):
        self.t2w_key = keys
    def __call__(self, data):
        d = dict(data)
        t2w =d[self.t2w_key]
        _,h,w = t2w.size()
        d['h'] = h
        d['w'] = w
        return d


def get_base_transforms(
    minv: int=0, 
    maxv: int=1
)->list:
    
    tfms=[]
    tfms+=[LoadImaged(keys=['t2w','adc','dwi','segmentation'], image_only=True)]
    tfms+=[EnsureChannelFirstd(keys=['t2w','adc','dwi','segmentation'])]
    tfms+=[SaveDim(keys='t2w')]
    tfms+=[
        ScaleIntensityd(
            keys=['t2w', 'adc', 'dwi'],
            minv=minv,
            maxv=maxv
        )]
    tfms+=[NormalizeIntensityd(keys=['t2w', 'adc', 'dwi'])]
    tfms += [CatImage(keys=['t2w','adc', 'dwi'])]
    
    return tfms

def get_train_transforms(cfg, p=0.175):
    tfms=get_base_transforms()
    if cfg.RandBiasFieldd>0:
        from monai.transforms import RandBiasFieldd
        tfms+=[
            RandBiasFieldd(
                keys=['image'],
                degree=10,
                coeff_range=[0.0, 0.01],
                prob=p
            )
        ]
    if cfg.RandGaussianSmoothd > 0:
        from monai.transforms import RandGaussianSmoothd
        tfms+=[
            RandGaussianSmoothd(
                keys=['image'],
                sigma_x= [0.25, 1.5],
                sigma_y= [0.25, 1.5],
                prob=p
            )
        ]
    if cfg.RandGibbsNoised >0:
        from monai.transforms import RandGibbsNoised
        tfms+=[
            RandGibbsNoised(
                keys=['image'],
                alpha=[0.5, 1],
                prob=p
            )
        ]

    if cfg.RandAffined >0:
        from monai.transforms import RandAffined
        tfms+=[
            RandAffined(
                keys=['image', 'segmentation'],
                rotate_range=5,
                shear_range=0.5,
                translate_range=25,
                mode="bilinear",
                prob=p
            )
        ]
    if cfg.RandRotate90d >0:
        from monai.transforms import RandRotate90d
        tfms+=[
            RandRotate90d(
                keys=['image', 'segmentation'],
                spatial_axes=[0,1],
                prob=p
            )
        ]
    if cfg.RandRotated >0:
        from monai.transforms import RandRotated
        tfms+=[
            RandRotated(
                keys=['image', 'segmentation'],
                range_x=0.1,
                range_y=0.1,
                mode=['bilinear', 'nearest'],
                prob=p
            )
        ]
    if cfg.RandElasticd >0:
        from monai.transforms import Rand2DElasticd as RandElasticd
        tfms+=[
            RandElasticd(
                keys=['image', 'segmentation'],
                spacing =[20,20],
                magnitude_range=[0.5, 1.5],
                rotate_range=5,
                shear_range=0.5,
                translate_range=25,
                mode=['bilinear', 'nearest'],
                prob=p
            )]
    if cfg.RandZoomd>0:
        from monai.transforms import RandZoomd
        tfms+=[
            RandZoomd(
                keys=['image', 'segmentation'],
                min_zoom=0.9,
                max_zoom=1.1,
                mode=['bilinear', 'nearest'],
                prob=p
            )]

    tfms+=[
        Resized(
            keys=['image', 'segmentation'],
            spatial_size=[384,384],
            mode = ['bilinear', 'nearest'],
        )
    ]
    from monai.transforms import RandSpatialCropd
    tfms+=[RandSpatialCropd(
            keys=['image', 'segmentation'],
            roi_size=[256,256],
            random_size=False,
            )
        ]        


    if cfg.RandGaussianNoised>0:
        from monai.transforms import RandGaussianNoised
        tfms+=[
            RandGaussianNoised(
                keys=['image'],
                mean=0.1,
                std=0.25,
                prob=p
            )
        ]

    if cfg.RandShiftIntensityd>0:
        from monai.transforms import RandShiftIntensityd
        tfms+=[
            RandShiftIntensityd(
                keys=['image'],
                offsets=0.2,
                prob=p
            )
        ]

    if cfg.RandGaussianSharpend>0:
        from monai.transforms import RandGaussianSharpend
        tfms+=[
            RandGaussianSharpend(
                keys=['image'],
                sigma1_x=[0.5, 1.0],
                sigma1_y=[0.5, 1.0],
                sigma2_x=[0.5, 1.0],
                sigma2_y=[0.5, 1.0],
                alpha=[10.0, 30.0],
                prob=p
            )
        ]

    if cfg.RandAdjustContrastd>0:
        from monai.transforms import RandAdjustContrastd
        tfms+=[
            RandAdjustContrastd(
                keys=['image'],
                gamma=2.0,
                prob=p
            )
        ]

    return Compose(tfms)


def get_val_transforms():
    tfms=get_base_transforms()
    tfms+=[
        Resized(
            keys=['image', 'segmentation'],
            spatial_size=[256,256],
            mode = ['bilinear', 'nearest'],
        )
    ]
    return Compose(tfms)


def get_test_transforms():
    tfms=get_base_transforms()

    tfms+=[
        Resized(
            keys=['image', 'segmentation'],
            spatial_size=[256,256],
            mode = ['bilinear', 'nearest'],
        )
    ]
  
    
    return Compose(tfms)

