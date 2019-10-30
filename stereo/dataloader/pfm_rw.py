#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import sys
import logging
logger = logging.getLogger(__name__)


def save_pfm(fname, image, scale=1):
    """
    save numpy array as a pfm file
    Args:
        fname: path to the file to be loaded
        image: a numpy array as image
        scale: scale of image
    Returns:
        None
    """
    file = open(fname, 'w') 
    color = None
     
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
     
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
     
    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))
     
    endian = image.dtype.byteorder
     
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
     
    file.write('%f\n' % scale)
     
    np.flipud(image).tofile(file)


def read_pfm(fname):
    """
    Load a pfm file as a numpy array
    Args:
        fname: path to the file to be loaded
    Returns:
        content of the file as a numpy array
    """
    file = open(fname, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if b'PF' == header:
        color = True
    elif b'Pf' == header:
        color = False
    else:
        raise Exception('Not a PFM file! header: ' + header)

    dims = file.readline()
    try:
        width, height = list(map(int, dims.split()))
    except:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    
    return data, scale


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
    
    import matplotlib.pyplot as plt
    import os
    image = np.float32(np.random.rand(256, 512, 3))
    image[64:128, 64:128] = 0
    fname = 'tmp.pfm'
    logger.info('Write pfm file ...')
    save_pfm(fname, image)
    logger.info('Read pfm file ...')
    image_read = read_pfm(fname)[0]
    plt.subplot(211);plt.imshow(image)
    plt.subplot(212);plt.imshow(image_read)
    plt.show()
    logger.info('Delete pfm file ...')
    os.unlink(fname)


