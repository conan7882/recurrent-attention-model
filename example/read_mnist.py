#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: read_mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import platform
import numpy as np
import scipy.ndimage.interpolation
sys.path.append('../')
from lib.dataflow.mnist import MNISTData

import matplotlib.pyplot as plt


if platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/MNIST_data/'
elif platform.node() == 'Qians-MacBook-Pro.local':
    pass
    # DATA_PATH = '/Users/gq/Google Drive/Foram/Training/simple_1/'
else:
    DATA_PATH = 'E:/Dataset/MNIST/'

def translate_image(im, im_size, trans_size):
    pad_border = int(np.ceil((trans_size - im_size) / 2.))
    pad_im = np.pad(im,
                    ((pad_border, pad_border), (pad_border, pad_border)),
                    'constant', constant_values=((0, 0), (0, 0)))
    translations = np.random.random(2) * 2. * pad_border - pad_border
    trans_im = scipy.ndimage.interpolation.shift(pad_im, translations)
    return trans_im

def translate_image_2(im, im_size, trans_size):
    pad_h = int(np.ceil((trans_size[0] - im_size[0]) / 2.))
    pad_w = int(np.ceil((trans_size[1] - im_size[1]) / 2.))
    pad_im = np.pad(im,
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    'constant', constant_values=((0, 0), (0, 0)))
    trans_h = np.random.random(1) * 2. * pad_h - pad_h
    trans_w = np.random.random(1) * 2. * pad_w - pad_w
    trans_im = scipy.ndimage.interpolation.shift(pad_im, (trans_h, trans_w))
    return trans_im

def put_two_image(im_1, im_2, im_size, canvas_size):
    pad_size = canvas_size - 2 * im_size
    assert pad_size >= 0
    divide_ratio = np.random.random(1)
    side_1 = int(im_size * 1. * divide_ratio)
    side_2 = im_size - side_1
    if np.random.random(1) > 0.5:
        trans_1 = translate_image_2(im_1, [im_size, im_size], [im_size, side_1])
        trans_2 = translate_image_2(im_2, [im_size, im_size], [im_size, side_2])
    else:
        trans_1 = translate_image_2(im_1, [im_size, im_size], [side_1, im_size])
        trans_2 = translate_image_2(im_2, [im_size, im_size], [side_2, im_size])


def translate_mnist(im):
    return translate_image(im, im_size=28, trans_size=100)


def trans_mnist(trans_size=0, batch_size=128):
    def translate_mnist(im):
        return translate_image(im, im_size=28, trans_size=trans_size)

    train_data = MNISTData('train',
                           data_dir=DATA_PATH,
                           shuffle=True,
                           pf=translate_mnist,
                           batch_dict_name=['im', 'label'])
    valid_data = MNISTData('val',
                           data_dir=DATA_PATH,
                           shuffle=True,
                           pf=translate_mnist,
                           batch_dict_name=['im', 'label'])

    train_data.setup(epoch_val=0, batch_size=batch_size)
    valid_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data, valid_data

