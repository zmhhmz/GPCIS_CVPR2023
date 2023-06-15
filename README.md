## Interactive Segmentation as Gaussian Process Classification (CVPR2023 Highlight)
Minghao Zhou, [Hong Wang](https://hongwang01.github.io/), Qian Zhao, Yuexiang Li, Yawen Huang, [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng), [Yefeng Zheng](https://sites.google.com/site/yefengzheng/)


[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Interactive_Segmentation_As_Gaussion_Process_Classification_CVPR_2023_paper.pdf) [[Poster]](https://cvpr2023.thecvf.com/media/PosterPDFs/CVPR%202023/23088.png?t=1684895990.406102) [[Video]](https://youtu.be/mapyH-WujhY) [[Slides]](https://cvpr2023.thecvf.com/media/cvpr-2023/Slides/23088.pdf) [[Supp]](GPCIS_supp.pdf)

## Usage
Please first set up the environment and prepare the training (SBD)/testing (GrabCut, Berkeley, SBD, DAVIS) datasets following [RITM](https://github.com/saic-vul/ritm_interactive_segmentation), and change the directories in [config.yml](config.yml). 

Please run [run.sh](run.sh) for training/evaluation. For training, the resnet50 weights pretrained on ImageNet is used. Please download the [weights](https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1s-1762acc0.pth) and change the corresponding directory in [config.yml](config.yml). For evaluation, you can directly test with our provided checkpoint in [checkpoints/GPCIS_Resnet50.pth](checkpoints/GPCIS_Resnet50.pth).

The core codes of the GPCIS model can be found in [isegm/model/is_gp_model.py](isegm/model/is_gp_model.py) and [isegm/model/is_gp_resnet50.py](isegm/model/is_gp_resnet50.py).

## Overview of GPCIS
<div  align="center"><img src="net.png" height="100%" width="100%" alt=""/></div>
