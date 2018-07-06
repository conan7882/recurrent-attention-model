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

def translate_mnist(im):
    return translate_image(im, im_size=28, trans_size=100)


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

# batch_data = train_data.next_batch_dict()

# print(translate_mnist(np.squeeze(batch_data['im'])))
# print(batch_data['im'])
# plt.figure()
# plt.imshow(np.squeeze(batch_data['im']))
# plt.show()

