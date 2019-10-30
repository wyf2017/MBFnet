#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import preprocess 
from dataset import dataset_by_name

import traceback
import logging
logger = logging.getLogger(__name__)

Test = False
preprocess_color = False

def rand():
    return torch.rand(1).item()


def randint(low, high):
    return int(torch.randint(low, high, [1, ]).item())


class RandomCrop(object):
    """Normalize image from the mean and std of samples
    >>> transform = RandomCrop([33, 34], [32, 32], training=True)
    >>> img = np.random.rand(33, 34, 3)
    >>> transform(img).shape
    (32, 32, 3)
    >>> transform = RandomCrop([29, 30], [32, 32], training=True)
    >>> img = np.random.rand(29, 30, 3)
    >>> transform(img).shape
    (32, 32, 3)
    """

    def __init__(self, input_size, crop_size, training=True):
        self.input_size = list(input_size)
        self.crop_size = list(crop_size)
        
        ih, iw = input_size
        ch, cw = crop_size
        
        # pad
        self.need_pad = (cw > iw) or (ch > ih)
        if(self.need_pad):
            ph = max(ch-ih, 0)
            pw = max(cw-iw, 0)
            self.pad = [(ph, 0), (pw, 0)]

        # rectangle [w1, h1, w2, h2]
        if training: 
            self.w1 = randint(0, iw - cw) if iw>cw else 0
            self.h1 = randint(0, ih - ch) if ih>ch else 0
        else:
            self.w1 = iw - cw if iw>cw else 0
            self.h1 = ih - ch if ih>ch else 0
        self.w2 = self.w1 + cw
        self.h2 = self.h1 + ch
        logger.debug('[w1, h1, w2, h2] : ' + str([self.w1, self.h1, self.w2, self.h2]))


    def __call__(self, img_numpy):
        
        if(img_numpy is None):
            return img_numpy
        
        assert list(img_numpy.shape[:2]) == self.input_size
        if(self.need_pad):
            pad = self.pad if 2==img_numpy.ndim else self.pad+[(0, 0)]
            img_numpy = np.pad(img_numpy, pad, 'constant')
        img_numpy = img_numpy[self.h1:self.h2, self.w1:self.w2]

        return img_numpy
        

class ImageFloder(data.Dataset):
    def __init__(self, datasets, n, training, crop_size):
 
        # self.n = 2/3/4
        # 2-->left, right 
        # 3-->left, right, disp_left 
        # 4-->left, right, disp_left, disp_right
        n = min([ds.num_in_group for ds in datasets] + [n])
        self.n = max(2, min(4, n)) 
        
        self.count_datasets = len(datasets)
        self.count = max([len(ds) for ds in datasets])
        self.datasets = datasets
        self.training = training

        # reset crop_size
        if (crop_size is not None) and ([0, 0]==list(crop_size)): # crop_size=[width, height]
            common_sizes = [ds.common_size for ds in datasets]
            if(None in common_sizes):
                crop_size = None
            elif(len(common_sizes)==1):
                crop_size = common_sizes[0]
            else:
                ws = [wh[0] for wh in common_sizes]
                hs = [wh[1] for wh in common_sizes]
                crop_size = [min(ws), min(hs)]
        self.crop_size = crop_size # [width, height]


    def _process(self):
        if preprocess_color: 
            return preprocess.get_transform(augment=self.training)
        else:
            return transforms.ToTensor()


    def _refill_invalid_disp(self, disp):

        if(disp is None):return None
        mask_valid = (disp>0)&(disp<10000)
        disp[~mask_valid] = 0 #-np.inf
        return disp


    def __getitem__(self, index):
        try:
            # get paths and loader and dploader
            idx = index % self.count_datasets
            paths_grp = self.datasets[idx][index//self.count_datasets]
            loader = self.datasets[idx].img_loader
            dploader = self.datasets[idx].disp_loader
            
            # loader image and disp
            tn = len(paths_grp)
            assert tn >= self.n
            filename = os.path.basename(paths_grp[0])
            if(Test): logger.info('load {} ...'.format(filename))
            left = np.array(loader(paths_grp[0]))
            right = np.array(loader(paths_grp[1]))
            disp_left = None
            disp_right = None

            # Random Horizontal Flip
            if(self.training and 0==tn%2 and rand()>0.5):
                left_t = np.fliplr(right).copy()
                right = np.fliplr(left).copy()
                left = left_t
                if(tn == 4): 
                    disp_left = np.fliplr(dploader(paths_grp[3])).copy()
                    if(self.n == 4):
                        disp_right = np.fliplr(dploader(paths_grp[2])).copy()
            
            else:
                if(self.n >= 3):
                    disp_left = np.ascontiguousarray(dploader(paths_grp[2]))
                if(self.n >= 4):
                    disp_right = np.ascontiguousarray(dploader(paths_grp[3]))

            # Random crop
            if self.crop_size is not None:
                crop_size = [self.crop_size[1], self.crop_size[0]]
                fun_crop = RandomCrop(left.shape[:2], crop_size, self.training)
                left = fun_crop(left)
                right = fun_crop(right)
                disp_left = fun_crop(disp_left)
                disp_right = fun_crop(disp_right)

            # preprocess
            process = self._process()
            left = process(left)
            right = process(right)
            disp_left = self._refill_invalid_disp(disp_left)
            disp_right = self._refill_invalid_disp(disp_right)

            # return
            if(Test): logger.info('{} loaded'.format(filename))
            if(self.n == 2):
                return filename, left, right
            elif(self.n == 3):
                return filename, left, right, disp_left[None]
            elif(self.n == 4):
                return filename, left, right, disp_left[None], disp_right[None]

        except Exception as err:
            logger.error(traceback.format_exc())
            msg = '[ Loadering data ] An exception happened: %s \n\t left: %s' % (str(err), paths_grp[0])
            logger.error(msg)
            index = randint(0, len(self)-1)
            return self.__getitem__(index)


    def __len__(self):
        return self.count*self.count_datasets


def dataloader_by_name(names='k2015-tr, k2012-tr', roots='./kitti, ./kitti', 
                        bn=1, training=False, crop_size=None, n=4):
    
    names = names.split(',')
    roots = roots.split(',')
    assert 1==len(roots) or len(names) == len(roots)
    
    datasets = [dataset_by_name(names[i], roots[i%len(roots)]) for i in range(len(names))]
    tImageFloder = ImageFloder(datasets, n, training, crop_size)
    if(tImageFloder.crop_size is None):
        bn = 1
    
#    dataloader = DataLoader(tImageFloder, batch_size=bn, shuffle=training, num_workers=bn, drop_last=False)
    dataloader = DataLoader(tImageFloder, batch_size=bn, shuffle=training, num_workers=2, drop_last=False)
    
    return dataloader


def batch_visualize(batch, name, normalize=True):
    
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    logger.info('batch[1].shape : ' + str(batch[1].shape))
    n = len(batch)

    imgs = torch.cat(batch[1:3], dim=0)
    #print('\n'.join(['max: {:.3f}, min: {:.3f}'.format(tmp.max(), tmp.min()) for tmp in imgs]))
    imgs = make_grid(imgs, nrow=4, padding=8, normalize=normalize).cpu()
    plt.subplot(2, 1, 1); plt.imshow(imgs.numpy().transpose(1, 2, 0))
    plt.title('%s( %s )' % (name, ','.join(batch[0])) )

    if(n > 3): 
        imgs = torch.cat(batch[3:], dim=0)
        imgs = make_grid(imgs, nrow=4, padding=8, normalize=False).cpu()
        plt.subplot(2, 1, 2); plt.imshow(imgs.numpy()[0])
        plt.title('min=%.2e' % imgs.min().item())

    plt.show() # plt.pause(0.1) # 


def test_dataloader_by_name(name, root, count=1):

    try:
        logger.info(' | '.join(['name: '+name, 'root: '+root]))
        training = ('-tr' in name)
        crop_size = [512, 256] if training else [0, 0]
        bn = 4
        dataloader = dataloader_by_name(name, root, bn, training, crop_size, 4)
        logger.info('count: %d' % len(dataloader))
        for batch_idx, batch in enumerate(dataloader):
            if(batch_idx>=count): break
            batch_visualize(batch, name)
        logger.info('passed!\n')
            
    except Exception as err:
        logger.error('\nAn exception happened! | Error message: %s \n' % (err))
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    
    import doctest
    doctest.testmod()

    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
    Test = True
    preprocess_color = True
    
    from dataset import datasets_dict
    datasets = datasets_dict()
#    datasets = []
#    import random
#    random.shuffle(datasets)
    
    # multi
    root = '/media/qjc/D/data/kitti'
    name_dataset = 'k15, k12'
    datasets.append({'root':root, 'name':name_dataset})
    
    for item in datasets:
        test_dataloader_by_name(item['name'], item['root'], count=40)


