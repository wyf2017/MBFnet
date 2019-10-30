#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import torch

import logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_setting():

    import argparse
    parser = argparse.ArgumentParser(description='Supervised Deep Stereo Matching by pytorch')
    parser.add_argument('--mode', default='train',
                        help='mode of execute [train/finetune/val/submission')
    
    # arguments of datasets
    parser.add_argument('--datas_train', default='k2015-tr, k2012-tr',
                        help='datasets for training')
    parser.add_argument('--datas_val', default='k2015-val, k2012-val',
                        help='datasets for validation')
    parser.add_argument('--dir_datas_train', default='/media/qjc/D/data/kitti/',
                        help='dirpath of datasets for training')
    parser.add_argument('--dir_datas_val', default='/media/qjc/D/data/kitti/',
                        help='dirpath of datasets for validation')
    parser.add_argument('--bn', type=int, default=4,
                        help='batch size')
    parser.add_argument('--crop_width', type=int, default=768,
                        help='width of crop_size')
    parser.add_argument('--crop_height', type=int, default=384,
                        help='height of crop_size')

    # arguments of model
    parser.add_argument('--arch', default='DispNetC',
                        help='select arch of model')
    parser.add_argument('--maxdisp', type=int ,default=192,
                        help='maxium disparity')
    parser.add_argument('--loadmodel', default=None, 
                        help='path of pretrained weight')

    # arguments of optimizer
    parser.add_argument('--freq_optim', type=int, default=1,
                        help='frequent of optimize weight')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learnig rate')
    parser.add_argument('--lr_epoch0', type=int, default=10,
                        help='the first epoch of adjust learnig rate')
    parser.add_argument('--lr_stride', type=int, default=3,
                        help='epoch stride of adjust learnig rate')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='decay factor of adjust learnig rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='decay factor of weight')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 of Adam')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 of Adam')

    # arguments for training
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--nloop', type=int, default=1,
                        help='loop of dataset in a epoch')
    parser.add_argument('--epochs_warmup', type=int, default=0,
                        help='number of epochs to warmup weight')
    parser.add_argument('--freq_save', type=int, default=1,
                        help='frequent of save weight')
    parser.add_argument('--freq_print', type=int, default=20,
                        help='frequent of print infomation')

    # other arguments
    parser.add_argument('--dir_save', default='./results/',
                        help='dirpath of save result( weight/submission )')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # parser arguments
    args = parser.parse_args()
    
    # add arguments
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.beta = (args.beta1, args.beta2)
    args.crop_size = (args.crop_width, args.crop_height)

    # log arguments
    items = sorted(args.__dict__.items())
    msg = 'The setted arguments as follow: \n'
    msg += '\n'.join([' [%s]: %s' % (k, str(v)) for k, v in items])
    logger.info(msg + '\n')

    return args


# program entry 
if __name__ == '__main__':

    # get setting
    args = get_setting()

    # set gpu id used
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # set manual seed
    if(not os.path.isdir(args.dir_save)):
        os.mkdir(args.dir_save)
    
    # excute stereo program
    import stereo
    
    if(args.mode.lower() in ['train', 'finetune']):
        stereo.train_val(args)
    
    elif(args.mode.lower() in ['val', 'validation']):
        stereo.val(args)
    
    elif(args.mode.lower() in ['sub', 'submission']):
        stereo.submission(args)

    else:
        logger.error('not support mode[ %s ]' % args.mode)


