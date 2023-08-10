# HFLIC
This section contains the official code for Human Friendly Perceptual Learned Image Compression with Reinforced Transform. Additionally, it includes an unofficial implementation of the paper titled "PO-ELIC: Perception-Oriented Efficient Learned Image Coding." The code implementation is based on CompressAI.
We share our enhance transform elic ckpt in [Enh-ELIC-ckpt](https://disk.pku.edu.cn:443/link/0C4548BF6A303EDBA16835CBC1405584), and modify the `config_5group.py` in `./config`, you can train HFLIC and Enh-POELIC with different lambda. We further share some of our code and lamda setting for ["EnhPO:ELIC" and "HFLIC"](https://disk.pku.edu.cn/#/link/5DB382C6C7D49AAC0AC2E4D7FE31E9A2)
# PO:ELIC
This section consists of an unofficial implementation of the paper titled "PO-ELIC: Perception-Oriented Efficient Learned Image Coding," which was presented at CVPR22W as the 1st place winner of CLIC22.
## How to train po:elic
To utilize the code effectively, follow these steps:
Open the file modules/layers/res_blk.py and modify the ResidualBottleneck class. Set N * 2 to N / 2.
Open the file config/config_5group.py and locate the line "lambda_face": 0. Modify this line to set the value of "lambda_face" as 0.
# ELIC
We have incorporated the ELIC code from the GitHub repository maintained by JiangWeibeta. You can find the code and related resources at the following link: https://github.com/JiangWeibeta/ELIC.

# Cite
## HFLIC
```
@misc{ning2023hflic,
      title={HFLIC: Human Friendly Perceptual Learned Image Compression with Reinforced Transform}, 
      author={Peirong Ning and Wei Jiang and Ronggang Wang},
      year={2023},
      eprint={2305.07519},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## PO:ELIC
```
@inproceedings{he2022po,
  title={PO-ELIC: Perception-Oriented Efficient Learned Image Coding},
  author={He, Dailan and Yang, Ziming and Yu, Hongjiu and Xu, Tongda and Luo, Jixiang and Chen, Yuan and Gao, Chenjian and Shi, Xinjie and Qin, Hongwei and Wang, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1764--1769},
  year={2022}
}
```
## ELIC
```
@misc{jiang2022unofficialelic,
    author={Jiang, Wei},
    title={Unofficial ELIC},
    howpublished={\url{https://github.com/JiangWeibeta/ELIC}},
    year={2022}
}
```

```
@inproceedings{he2022elic,
  title={Elic: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding},
  author={He, Dailan and Yang, Ziming and Peng, Weikun and Ma, Rui and Qin, Hongwei and Wang, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5718--5727},
  year={2022}
}
