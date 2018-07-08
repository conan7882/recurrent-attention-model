#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import read_mnist as read
# 
train_data, valid_data = read.two_digits_mnist(canvas_size=100, batch_size=1)
# train_data, valid_data = read.trans_mnist(trans_size=100, batch_size=10)

batch_data = train_data.next_batch_dict()
print(np.array(batch_data['im']).shape)
print(batch_data['label'])
print(valid_data.size())
print(valid_data.im_list.shape[0])

plt.figure()
plt.imshow(np.squeeze(batch_data['im']))
plt.show()