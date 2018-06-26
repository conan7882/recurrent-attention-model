#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf 


def Linear(inputs, out_dim, name='Linear', nl=tf.identity):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        inputs = batch_flatten(inputs)
        in_dim = inputs.get_shape().as_list()[1]
        weights = tf.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  # dtype=None,
                                  initializer=None,
                                  regularizer=None,
                                  trainable=True)
        biases = tf.get_variable('biases',
                                  shape=[out_dim],
                                  # dtype=None,
                                  initializer=None,
                                  regularizer=None,
                                  trainable=True)
        print('init: {}'.format(weights))
        act = tf.nn.xw_plus_b(inputs, weights, biases)

        return nl(act, name='output')


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))
