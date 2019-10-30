#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import abc
import os
import glob
import pickle
import numpy as np
from PIL import Image
from pfm_rw import read_pfm

import logging
logger = logging.getLogger(__name__)


class dataset_stereo(object):
    '''stereo dataset'''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, root):
        # name, root, paths of dataset
        self.name = None
        self.root = root
        self.n_root = len(self.root)
        self.dir_save = os.path.join(self.root, 'paths')
        
        # count and paths of dataset
        self.keys = ['img_left', 'img_right', 'disp_left', 'disp_right']
        self.n = 2
        self.paths = []
        self.count = 0

        # flag of convert path from left to others
        self.str_filter = None
        self.flag_img_left_right = [None, None]
        self.flag_img_disp_left = [None, None]
        self.flag_disp_left_right = [None, None]
        self.flag_img_type = None
        self.flag_disp_type = None
        self.flag_same_type = True
        
    @property
    def num_in_group(self):
        return self.n
    
    @abc.abstractproperty
    def common_size(self):
        #return [width, height]
        raise NotImplementedError
    
    @abc.abstractmethod
    def disp_loader(self, path):
        #return disp[numpy.array]
        raise NotImplementedError
        
    def img_loader(self, path, fliplr=False, flipud=False):
        
        return Image.open(path).convert('RGB')


    def append(self, paths_group):
        self.paths.append(paths_group)
        self.count += 1
    
    def extend(self, list_groups):
        self.paths.extend(list_groups)
        self.count += len(list_groups)
    
    def get_paths_all(self, str_filter_glob, flag_sort=False):
        file_paths = os.path.join(self.dir_save, self.name+'.pkl')
        if(os.path.isfile(file_paths)):
            with open(file_paths,'rb') as f:
                self.extend(pickle.load(f))
            # 检查数据集是否为空
            assert self.count>0, 'dataset[%s] do not exist!' % (self.name)
        else:
            self.get_paths_all_filter(str_filter_glob, flag_sort)
            if(not os.path.isdir(self.dir_save)):
                os.mkdir(self.dir_save)
            with open(file_paths,'wb') as f:
                pickle.dump(self.paths, f, True)

   
    def get_paths_all_filter(self, str_filter_glob, flag_sort=False):
        # 获取所有左图片路径
        logger.debug('str_filter_glob:' + str_filter_glob)
        paths_img_left = glob.glob(str_filter_glob)
        if(flag_sort): paths_img_left.sort()
        logger.debug('根据str_filter_glob得到的文件个数：%d' % len(paths_img_left))
        
        # 根据左图片路径获取图片组的路径
        for path_img_left in paths_img_left:
            paths_group = self.get_group_from_left(path_img_left)
            self.append(paths_group)
        
        # 检查数据集是否为空
        assert self.count>0, 'dataset[%s] do not exist!' % (self.name)

   
    def get_group_from_left(self, path_img_left):
        paths = []
        # 防止root中存在干扰字符串
        path_img_left = path_img_left[self.n_root:]
        paths.append(self.root + path_img_left)
        
        # path_img_right
        path_img_right = path_img_left.replace(*self.flag_img_left_right)
        paths.append(self.root + path_img_right)
        
        # path_disp_left
        if(self.flag_img_disp_left[0] is None):
            return paths
        path_disp_left = path_img_left.replace(*self.flag_img_disp_left)
        if(not self.flag_same_type):
            path_disp_left = path_disp_left.replace(self.flag_img_type, self.flag_disp_type)
        paths.append(self.root + path_disp_left)
        
        # path_disp_right
        if(self.flag_disp_left_right[0] is None):
            return paths
        path_disp_right = path_disp_left.replace(*self.flag_disp_left_right)
        paths.append(self.root + path_disp_right)
        
        return paths


    def __getitem__(self, idx):
        assert self.count>0, 'dataset[%s] do not exist!' % (self.name)
        idx %= self.count
        out = self.paths[idx]
        return out

    def __len__(self):
        return self.count

class dataset_sceneflow(dataset_stereo):
    '''dataset_sceneflow'''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, root):
        super(dataset_sceneflow, self).__init__(root)
        
        # flag of convert path from left to others
        self.flag_img_left_right = ['left', 'right']
        self.flag_img_disp_left = ['frames_finalpass_webp', 'disparity']
        self.flag_disp_left_right = ['left', 'right']
        self.flag_img_type = '.webp'
        self.flag_disp_type = '.pfm'
        self.flag_same_type = (self.flag_img_type==self.flag_disp_type)

    @property
    def common_size(self):
        return [960, 540]
    
    def disp_loader(self, path):
        disp, scale = read_pfm(path)
        return disp


class dataset_fly3d_train(dataset_sceneflow):
    '''dataset_fly3d_train'''

    def __init__(self, root):
        super(dataset_fly3d_train, self).__init__(root)
        self.name = 'fly3d-tr'

        # get_paths_all
        flag = 'flyingthings3d/frames_finalpass_webp/TRAIN/*/*/left/*.webp'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=False)

class dataset_fly3d_test(dataset_sceneflow):
    '''dataset_fly3d_test'''
    def __init__(self, root):
        super(dataset_fly3d_test, self).__init__(root)
        self.name = 'fly3d-te'

        # get_paths_all
        flag = 'flyingthings3d/frames_finalpass_webp/TEST/*/*/left/*.webp'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=False)

class dataset_monkaa(dataset_sceneflow):
    '''dataset_monkaa'''
    def __init__(self, root):
        super(dataset_monkaa, self).__init__(root)
        self.name = 'monkaa'

        # get_paths_all
        flag = 'monkaa/frames_finalpass_webp/*/left/*.webp'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=False)

class dataset_driving(dataset_sceneflow):
    '''dataset_driving'''
    def __init__(self, root):
        super(dataset_driving, self).__init__(root)
        self.name = 'driving'

        # get_paths_all
        flag = 'driving/frames_finalpass_webp/*/*/*/left/*.webp'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=False)

class dataset_sceneflow_train(dataset_sceneflow):
    '''dataset_sceneflow_train'''
    def __init__(self, root):
        super(dataset_sceneflow_train, self).__init__(root)
        self.name = 'sceneflow-tr'
        dataset1 = dataset_fly3d_train(root)
        dataset2 = dataset_monkaa(root)
        dataset3 = dataset_driving(root)
        self.extend(dataset1.paths)
        self.extend(dataset2.paths)
        self.extend(dataset3.paths)


class dataset_kitti(dataset_stereo):
    '''dataset_kitti'''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, root):
        super(dataset_kitti, self).__init__(root)

        # flag of convert path from left to others
        self.flag_img_left_right = [None, None]
        self.flag_img_disp_left = [None, None]
        self.flag_disp_left_right = [None, None]
        self.flag_img_type = '.png'
        self.flag_disp_type = '.png'
        self.flag_same_type = (self.flag_img_type==self.flag_disp_type)

    @property
    def common_size(self):
        return [1226, 352]
    
    def disp_loader(self, path):
        disp = Image.open(path)
        disp = np.array(disp, dtype=np.float32)/256.0
        return disp


class dataset_kitti_raw(dataset_kitti):
    '''dataset_kitti_raw'''
    def __init__(self, root):
        super(dataset_kitti_raw, self).__init__(root)
        self.name = 'kitti-raw'
        
        # flag of convert path from left to others
        self.flag_img_left_right = ['image_02', 'image_03']

        # get_paths_all
        flag = 'raw/*/*/image_02/data/*.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=False)

class dataset_kitti2015_train(dataset_kitti):
    '''dataset_kitti2015_train'''
    def __init__(self, root):
        super(dataset_kitti2015_train, self).__init__(root)
        self.name = 'kitti2015-tr'

        # flag of convert path from left to others
        self.flag_img_left_right = ['image_2', 'image_3']
        self.flag_img_disp_left = ['image_2', 'disp_occ_0']

        # get_paths_all
        flag = 'data_scene_flow/training/image_2/*_10.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

class dataset_kitti2015_test(dataset_kitti):
    '''dataset_kitti2015_test'''
    def __init__(self, root):
        super(dataset_kitti2015_test, self).__init__(root)
        self.name = 'kitti2015-te'

        # flag of convert path from left to others
        self.flag_img_left_right = ['image_2', 'image_3']

        # get_paths_all
        flag = 'data_scene_flow/testing/image_2/*_10.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

class dataset_kitti2012_train(dataset_kitti):
    '''dataset_kitti2012_train'''
    def __init__(self, root):
        super(dataset_kitti2012_train, self).__init__(root)
        self.name = 'kitti2012-tr'

        # flag of convert path from left to others
        self.flag_img_left_right = ['colored_0', 'colored_1']
        self.flag_img_disp_left = ['colored_0', 'disp_occ']

        # get_paths_all
        flag = 'data_stereo_flow/training/colored_0/*_10.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

class dataset_kitti2012_test(dataset_kitti):
    '''dataset_kitti2012_test'''
    def __init__(self, root):
        super(dataset_kitti2012_test, self).__init__(root)
        self.name = 'kitti2012-te'

        # flag of convert path from left to others
        self.flag_img_left_right = ['colored_0', 'colored_1']

        # get_paths_all
        flag = 'data_stereo_flow/testing/colored_0/*_10.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

class dataset_middeval(dataset_stereo):
    '''dataset_middeval'''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, root):
        super(dataset_middeval, self).__init__(root)
        
        # flag of convert path from left to others
        self.flag_img_left_right = ['im0.png', 'im1.png']
        self.flag_img_disp_left = [None, None]
        self.flag_disp_left_right = [None, None]
        self.flag_img_type = None
        self.flag_disp_type = None
        self.flag_same_type = (self.flag_img_type==self.flag_disp_type)

    @property
    def common_size(self):
        return None
    
    def disp_loader(self, path):
        disp, scale = read_pfm(path)
        return disp


class dataset_middeval_train(dataset_middeval):
    '''dataset_middeval_train'''

    def __init__(self, root):
        super(dataset_middeval_train, self).__init__(root)
        self.name = 'middeval-tr'

        # flag of convert path from left to others
        self.flag_img_left_right = ['im0.png', 'im1.png']
        self.flag_img_disp_left = ['im0.png', 'disp0GT.pfm']
        self.flag_disp_left_right = ['disp0GT.pfm', 'disp1GT.pfm']

        # get_paths_all
        flag = 'training*/*/im0.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

class dataset_middeval_test(dataset_middeval):
    '''dataset_middeval_test'''

    def __init__(self, root):
        super(dataset_middeval_test, self).__init__(root)
        self.name = 'middeval-te'

        # flag of convert path from left to others
        self.flag_img_left_right = ['im0.png', 'im1.png']

        # get_paths_all
        flag = 'test*/*/im0.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

class dataset_eth3d_train(dataset_middeval):
    '''dataset_eth3d_train'''

    def __init__(self, root):
        super(dataset_eth3d_train, self).__init__(root)
        self.name = 'eth3d-tr'

        # flag of convert path from left to others
        self.flag_img_left_right = ['im0.png', 'im1.png']
        self.flag_img_disp_left = ['im0.png', 'disp0GT.pfm']

        # get_paths_all
        flag = 'two_view_training/*/im0.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

class dataset_eth3d_test(dataset_middeval):
    '''dataset_eth3d_test'''

    def __init__(self, root):
        super(dataset_eth3d_test, self).__init__(root)
        self.name = 'eth3d-te'

        # flag of convert path from left to others
        self.flag_img_left_right = ['im0.png', 'im1.png']

        # get_paths_all
        flag = 'two_view_test/*/im0.png'
        self.str_filter = os.path.join(root, flag)
        self.get_paths_all(self.str_filter, flag_sort=True)

def dataset_by_name(name='kitti2015-tr', root='./kitti'):
    '''dataset_by_name'''

    root = root.strip()
    name = name.lower().strip()
    name = name.replace('kitti', 'k')
    name = name.replace('flyingthings3d', 'fly3d')
    name = name.replace('sceneflow', 'sf')

    if(name in ['k-raw']):
        n = 2
        dataset = dataset_kitti_raw(root)
        paths = dataset.paths[:]

    elif(name in ['k2012-te', 'k12-te']):
        n = 2
        dataset = dataset_kitti2012_test(root)
        paths = dataset.paths[:]

    elif(name in ['k2012', 'k12']):
        n = 3
        dataset = dataset_kitti2012_train(root)
        paths = dataset.paths[:]

    elif(name in ['k2012-tr', 'k12-tr']):
        n = 3
        dataset = dataset_kitti2012_train(root)
        paths = dataset.paths[:160]

    elif(name in ['k2012-val', 'k12-val']):
        n = 3
        dataset = dataset_kitti2012_train(root)
        paths = dataset.paths[160:]

    elif(name in ['k2015-te', 'k15-te']):
        n = 2
        dataset = dataset_kitti2015_test(root)
        paths = dataset.paths[:]

    elif(name in ['k2015', 'k15']):
        n = 3
        dataset = dataset_kitti2015_train(root)
        paths = dataset.paths[:]

    elif(name in ['k2015-tr', 'k15-tr']):
        n = 3
        dataset = dataset_kitti2015_train(root)
        paths = dataset.paths[:160]

    elif(name in ['k2015-val', 'k15-val']):
        n = 3
        dataset = dataset_kitti2015_train(root)
        paths = dataset.paths[160:]

    elif(name in ['sf-tr']):
        n = 4
        dataset = dataset_sceneflow_train(root)
        paths = dataset.paths[:]

    elif(name in ['sf-val', 'sf-te', 'fly3d-val', 'fly3d-te']):
        n = 4
        dataset = dataset_fly3d_test(root)
        paths = dataset.paths[:]

    elif(name in ['fly3d-tr']):
        n = 4
        dataset = dataset_fly3d_train(root)
        paths = dataset.paths[:]

    elif(name in ['driving']):
        n = 4
        dataset = dataset_driving(root)
        paths = dataset.paths[:]

    elif(name in ['monkaa']):
        n = 4
        dataset = dataset_monkaa(root)
        paths = dataset.paths[:]

    elif(name in ['middeval-te']):
        n = 2
        dataset = dataset_middeval_test(root)
        paths = dataset.paths[:]

    elif(name in ['middeval']):
        n = 4
        dataset = dataset_middeval_train(root)
        paths = dataset.paths[:]

    elif(name in ['middeval-tr']):
        n = 4
        dataset = dataset_middeval_train(root)
        paths = []
        for i in range(1, 5):
            paths = paths+dataset.paths[i::5]

    elif(name in ['middeval-val']):
        n = 4
        dataset = dataset_middeval_train(root)
        paths = dataset.paths[::5]

    elif(name in ['eth3d-te']):
        n = 2
        dataset = dataset_eth3d_test(root)
        paths = dataset.paths[:]

    elif(name in ['eth3d']):
        n = 3
        dataset = dataset_eth3d_train(root)
        paths = dataset.paths[:]

    elif(name in ['eth3d-tr']):
        n = 3
        dataset = dataset_eth3d_train(root)
        paths = []
        for i in range(1, 5):
            paths = paths+dataset.paths[i::5]

    elif(name in ['eth3d-val']):
        n = 3
        dataset = dataset_eth3d_train(root)
        paths = dataset.paths[::5]

    else:
        msg = '暂不支持的数据集: \n '
        msg = '目前只支持以下数据集: \n '
        msg += 'k-raw \n '
        msg += 'k2012-tr \n k2012-val \n k2012-te \n k2012 \n '
        msg += 'k2015-tr \n k2015-val \n k2015-te \n k2015 \n '
        msg += 'k12-tr \n k12-val \n k12-te \n k12 \n '
        msg += 'k15-tr \n k15-val \n k15-te \n k15 \n '
        msg += 'sf-tr \n sf-val \n sf-te \n '
        msg += 'fly3d-tr \n fly3d-val \n fly3d-te \n '
        msg += 'driving \n monkaa \n '
        msg += 'middeval-tr \n middeval-val \n middeval-te \n middeval \n '
        msg += 'eth3d-tr \n eth3d-val \n eth3d-te \n eth3d \n '
        msg += '请检查数据集名称和根路径! \n '
        msg += 'name: %s | root: %s \n ' % (name, root)
        logger.info(msg)
        return None

    dataset.n = n
    dataset.paths = paths
    dataset.count = len(paths)
    return dataset


def datasets_dict():

    datasets = [] # 需要测试的所有数据集
    root_data = '/media/qjc/D/data'

    # kitti
    root = os.path.join(root_data, 'kitti')
    for dataset in ['k15', 'k12']:
        for subfix in ['-tr', '-val', '-te', '']:
            name_dataset = dataset+subfix
            datasets.append({'root':root, 'name':name_dataset})
    datasets.append({'root':root, 'name':'kitti-raw'})

    # sceneflow
    root = os.path.join(root_data, 'sceneflow')
    for subfix in ['-tr', '-val', '-te']:
        name_dataset = 'sf'+subfix
        datasets.append({'root':root, 'name':name_dataset})

    for name_dataset in ['fly3d-tr', 'fly3d-te', 'monkaa', 'driving']:
        datasets.append({'root':root, 'name':name_dataset})

    # middeval
    root = os.path.join(root_data, 'MiddEval3')
    for subfix in ['-tr', '-val', '-te', '']:
        name_dataset = 'middeval'+subfix
        datasets.append({'root':root, 'name':name_dataset})

    # eth3d
    root = os.path.join(root_data, 'eth3d')
    for subfix in ['-tr', '-val', '-te', '']:
        name_dataset = 'eth3d'+subfix
        datasets.append({'root':root, 'name':name_dataset})
    
    return datasets


def test_dataset_by_name(name, root):
    try:
        dataset = dataset_by_name(name, root)
        count = len(dataset)
        n = dataset.n
        paths = dataset[count//2]
        logger.info('\n '.join(['name: '+name, 'root: '+root, 'count: %d' % count, 'n: %d' % n] + 
                                ['paths-%d' % (count//2)] + paths + ['passed!\n']))

    except Exception as err:
        logger.warning('\n An exception happened! | Error message: %s \n' % (err))


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
    
    for item in datasets_dict():
        test_dataset_by_name(item['name'], item['root'])


