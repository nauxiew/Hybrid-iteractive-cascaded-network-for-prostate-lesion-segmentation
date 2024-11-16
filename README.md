# Hybrid-iteractive-cascaded-network-for-prostate-lesion-segmentation
Here is the pytorch implementation of our iterative cascaded network for prostate lesion segmentation with automated quality assessment (https://www.mdpi.com/2306-5354/11/8/796) 

# Preparation
The pre-trained model of the backbone used in this work can be downloaded from [this link](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth)

# Train & Test
The training and test of fine segmentation network can refer to [link1](https://github.com/facebookresearch/segment-anything) and [link2](https://github.com/bowang-lab/MedSAM/tree/main)
# Reference
The citation detail of our method:
```bibtex
@article{kou2024interactive,
  title={Interactive Cascaded Network for Prostate Cancer Segmentation from Multimodality MRI with Automated Quality Assessment},
  author={Kou, Weixuan and Rey, Cristian and Marshall, Harry and Chiu, Bernard},
  journal={Bioengineering},
  volume={11},
  number={8},
  pages={796},
  year={2024},
  publisher={MDPI}
}

