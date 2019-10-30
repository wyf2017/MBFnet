# Stereo Datasets

1. [SceneFlow](#SceneFlow)
2. [KITTI](#KITTI)
3. [MiddEval3](#MiddEval3)
4. [ETH3d](#ETH3d)

##SceneFlow

* SceneFlow includes three datasets: Flyingthing3d, Driving and Monkaa.
* Download the dataset from [this](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). And unzip them to corresponding folder.
* the following is the describtion of 6 subfolder.
```
    # the disp and image folder of Flyingthing3d dataset
    [root]/flyingthings3d/disparity  
    [root]/flyingthings3d/frames_finalpass_webp  

    # the disp and image folder of Driving dataset
    [root]/driving/disparity  
    [root]/driving/frames_finalpass_webp  

    # the disp and image folder of Monkaa dataset
    [root]/monkaa/disparity  
    [root]/monkaa/frames_finalpass_webp  
```

##KITTI

* KITTI includes three datasets: KITTI2015 and KITTI2012
* Download the dataset from [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo). And unzip them to corresponding folder.
* the following is the describtion of 10 subfolder.
```
    # the disp and image folder of KITTI2015
    [root]/data_scene_flow/training/disp_occ_0  
    [root]/data_scene_flow/training/image_2  
    [root]/data_scene_flow/training/image_3  
    [root]/data_scene_flow/testing/image_2  
    [root]/data_scene_flow/testing/image_3  

    # the disp and image folder of KITTI2012
    [root]/data_stereo_flow/training/disp_occ  
    [root]/data_stereo_flow/training/colored_0  
    [root]/data_stereo_flow/training/colored_1  
    [root]/data_stereo_flow/testing/colored_0  
    [root]/data_stereo_flow/testing/colored_1  
```

##MiddEval3

* Download the dataset from [MiddEval3](http://). And unzip them to corresponding folder.
* the following is the describtion of subfolder.
```
    # the disp and image folder of MiddEval3
    [root]/training*/*/disp0GT.pfm
    [root]/training*/*/im0.png
    [root]/training*/*/im1.png
    [root]/test*/*/disp0GT.pfm
    [root]/test*/*/im0.png
    [root]/test*/*/im1.png
```

##ETH3d

* Download the dataset from [ETH3d](http://). And unzip them to corresponding folder.
* the following is the describtion of subfolder.
```
    # the disp and image folder of MiddEval3
    [root]/two_view_training/*/disp0GT.pfm
    [root]/two_view_training/*/im0.png
    [root]/two_view_training/*/im1.png
    [root]/two_view_test/*/disp0GT.pfm
    [root]/two_view_test/*/im0.png
    [root]/two_view_test/*/im1.png
```

