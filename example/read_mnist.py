#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: read_mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import platform
import numpy as np
import scipy.ndimage.interpolation
sys.path.append('../')
from lib.dataflow.mnist import MNISTData, MNISTPair


if platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/MNIST_data/'
elif platform.node() == 'Qians-MacBook-Pro.local':
    DATA_PATH = '/Users/gq/workspace/Dataset/MNIST_data/'
    # DATA_PATH = '/Users/gq/Google Drive/Foram/Training/simple_1/'
else:
    DATA_PATH = 'E:/Dataset/MNIST/'

def translate_image(im, im_size, trans_size):
    # assume square original image and transform image
    pad_border_1 = int(np.floor((trans_size - im_size) / 2.))
    pad_border_2 = trans_size - im_size - pad_border_1
    pad_im = np.pad(im,
                    ((pad_border_1, pad_border_2), (pad_border_1, pad_border_2)),
                    'constant', constant_values=((0, 0), (0, 0)))
    translations = np.random.random(2) * 2. * pad_border_1 - pad_border_1
    trans_im = scipy.ndimage.interpolation.shift(pad_im, translations)
    return trans_im

def translate_image_2(im, im_size, trans_size):
    # non-square original image and transform image
    pad_h_1 = int(np.floor((trans_size[0] - im_size[0]) / 2.))
    pad_h_2 = trans_size[0] - im_size[0] - pad_h_1
    pad_w_1 = int(np.floor((trans_size[1] - im_size[1]) / 2.))
    pad_w_2 = trans_size[1] - im_size[1] - pad_w_1
    pad_im = np.pad(im,
                    ((pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                    'constant', constant_values=((0, 0), (0, 0)))
    trans_h = np.random.random(1) * 2. * pad_h_1 - pad_h_1
    trans_w = np.random.random(1) * 2. * pad_w_1 - pad_w_1
    trans_im = scipy.ndimage.interpolation.shift(pad_im, (trans_h, trans_w))
    return trans_im

def put_two_image(im_1, im_2, im_size, canvas_size):
    pad_size = canvas_size - 2 * im_size
    assert pad_size >= 0
    divide_ratio = np.random.random(1)
    side_1 = int(pad_size * 1. * divide_ratio + im_size)
    side_2 = canvas_size - side_1

    # print(divide_ratio, side_1, side_2)
    if np.random.random(1) > 0.5:
        trans_1 = translate_image_2(im_1, [im_size, im_size], [canvas_size, side_1])
        trans_2 = translate_image_2(im_2, [im_size, im_size], [canvas_size, side_2])
        if np.random.random(1) > 0.5:
            return np.concatenate((trans_1, trans_2), axis=-1)
        else:
            return np.concatenate((trans_2, trans_1), axis=-1)
    else:
        trans_1 = translate_image_2(im_1, [im_size, im_size], [side_1, canvas_size])
        trans_2 = translate_image_2(im_2, [im_size, im_size], [side_2, canvas_size])
        if np.random.random(1) > 0.5:
            return np.concatenate((trans_1, trans_2), axis=0)
        else:
            return np.concatenate((trans_2, trans_1), axis=0) 


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

def two_digits_mnist(canvas_size, batch_size=128):
    def two_digits(im_1, im_2):
        return put_two_image(im_1, im_2, im_size=28, canvas_size=canvas_size)

    label_dict = {}
    label_id = 0
    for i in range(0, 10):
        for j in range(0, 10):
            if '{}{}'.format(i, j) not in label_dict:
                label_dict['{}{}'.format(i, j)] = label_id
                label_dict['{}{}'.format(j, i)] = label_id
                label_id += 1

    train_data = MNISTPair('train',
                           data_dir=DATA_PATH,
                           shuffle=True,
                           label_dict=label_dict,
                           batch_dict_name=['im', 'label'],
                           pairprocess=two_digits)
    valid_data = MNISTPair('val',
                           data_dir=DATA_PATH,
                           shuffle=True,
                           label_dict=label_dict,
                           batch_dict_name=['im', 'label'],
                           pairprocess=two_digits)

    train_data.setup(epoch_val=0, batch_size=batch_size)
    valid_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data, valid_data



