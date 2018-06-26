#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

# from tensorflow.examples.tutorials.mnist import input_data
from tensorcv.dataflow.dataset.MNIST import MNISTLabel


class MNISTData(MNISTLabel):
    def next_batch_dict(self):
        batch_data = self.next_batch()
        return {'data': batch_data[0], 'label': batch_data[1]}


