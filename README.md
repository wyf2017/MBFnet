# Pytorch Implementation of the Real-time Stereo Matching with a Hierarchical Refinement

citation{
Wang Yufeng,Wang Hongwei,Yu Guang,Yang Mingquan,Yuan Yuwei,Quan Jicheng. Real-time stereo matching with a hierarchical refinement [J]. Acta Optica Sinica, 2020, 40(09): 0915002.
王玉锋,王宏伟,于光,杨明权,袁昱纬,全吉成. 渐进细化的实时立体匹配算法[J]. 光学学报, 2020, 40(09): 0915002.
}

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

The stereo matching based on convolutional neural network has achieved a great improvement in accuracy, but the most still cannot meet the real-time requirements. A real-time stereo matching with a coarse-to-fine fashion is proposed, which initializes the disparity map at the low-resolution level and gradually restores the spatial resolution of the disparity map. The algorithm uses a lightweight backbone network to extract multi-scale features, and at the same time the features are inversely fused to improve the robustness of the features while ensuring the real-time performance. A multi-branch fusion(MBF) module is proposed to progressively refine the disparity map. The multi-modes of different regions are automatically clustered and processed separately, and then the final result is combined according to the cluster weights, so that the regions with different characteristic can be better processed. 

## Usage

### Dependencies

- [Python-3.7](https://www.python.org/downloads/)
- [PyTorch-1.1.0](http://pytorch.org)
- [torchvision-0.4.0](http://pytorch.org)
- [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [SceneFlow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Usage of KITTI and SceneFlow dataset in [stereo/dataloader/README.md](stereo/dataloader/README.md)

### Train
Reference to [demos/train_sfk_all.sh](demos/train_sfk_all.sh) and [demos/train_sfk.sh](demos/train_sfk.sh).

### Submission
Reference to [demos/submission_all.sh](demos/submission_all.sh) and [demos/submission.sh](demos/submission.sh).

[Pretrained Model](https://pan.baidu.com/s/1itwxOxwzgM0Rsk93sEpwIQ)

## Results
Reults on [KITTI 2015 leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |Environment|
|---|---|---|---|---|
| [DeepPruner(fast)](http://arxiv.org/abs/1909.05845) | 2.59 % | 2.35 % | 0.06 | Titan XP (Caffe) |
| [MBFnet(our)]() | 2.96 % | 2.54 % | 0.05 | 2070 (pytorch) |
| [DispNetC](http://arxiv.org/abs/1512.02134) | 4.32 % | 4.05 % | 0.06 | Titan XP (Caffe) |
| [MADnet](http://arxiv.org/abs/1810.05424) | 4.66 % | 4.27 % | 0.02 | 1080Ti (tensorflow) |


## Contacts
wangyf_1991@163.com

Any discussions or concerns are welcomed!
