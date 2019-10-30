# Pytorch Implementation of the Real-time Stereo Matching with a Hierarchical Refinement

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

The stereo matching based on convolutional neural network has achieved a great improvement in accuracy, but the most still cannot meet the real-time requirements. A real-time stereo matching with a coarse-to-fine fashion is proposed, which initializes the disparity map at the low-resolution level and gradually restores the spatial resolution of the disparity map. The algorithm uses a lightweight backbone network to extract multi-scale features, and at the same time the features are inversely fused to improve the robustness of the features while ensuring the real-time performance. A multi-branch fusion(MBF) module is proposed to progressively refine the disparity map. The multi-modes of different regions are automatically clustered and processed separately, and then the final result is combined according to the cluster weights, so that the regions with different characteristic can be better processed. 


## Contacts
wangyf_1991@163.com

Any discussions or concerns are welcomed!
