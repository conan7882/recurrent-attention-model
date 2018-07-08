#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tensorcv.models.layers import *
import tensorcv.models.layers as layers


def softmax(inputs):
    max_in = tf.reduce_max(inputs, axis=-1)
    max_in = tf.tile(tf.reshape(max_in, (-1, 1)), [1, inputs.shape[-1]])
    stable_in = inputs - max_in
    normal_p = tf.reduce_sum(tf.exp(stable_in), axis=-1)
    normal_p = tf.tile(tf.reshape(normal_p, (-1, 1)), [1, inputs.shape[-1]])
    return tf.exp(stable_in) / normal_p

def softplus(inputs):
    return tf.log(1 + tf.exp(inputs))

def linear(out_dim,
           layer_dict=None,
           inputs=None,
           init_w=None,
           init_b=tf.zeros_initializer(),
           name='Linear',
           nl=tf.identity):
    # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name):
        if inputs is None:
            assert layer_dict is not None
            inputs = layer_dict['cur_input']
        inputs = layers.batch_flatten(inputs)
        in_dim = inputs.get_shape().as_list()[1]
        weights = tf.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  # dtype=None,
                                  initializer=init_w,
                                  regularizer=None,
                                  trainable=True)
        biases = tf.get_variable('biases',
                                  shape=[out_dim],
                                  # dtype=None,
                                  initializer=init_b,
                                  regularizer=None,
                                  trainable=True)
        # print('init: {}'.format(weights))
        act = tf.nn.xw_plus_b(inputs, weights, biases)
        result = nl(act, name='output')
        if layer_dict is not None:
            layer_dict['cur_input'] = result
            
        return result

def make_cell(hidden_size, forget_bias=0.0,
              is_training=True, keep_prob=1.0):

    cell = tf.contrib.rnn.LSTMCell(
        num_units=hidden_size,
        use_peepholes=False,
        cell_clip=None,
        initializer=None,
        num_proj=None,
        proj_clip=None,
        num_unit_shards=None,
        num_proj_shards=None,
        forget_bias=forget_bias,
        state_is_tuple=True,
        activation=None,
        reuse=None,
        name='lstm'
    )

    if is_training is True:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, 
            output_keep_prob=keep_prob,
            variational_recurrent=True,
            dtype=tf.float32)
    return cell

def instance_norm(input, name="instance_norm"):
    # borrow from 
    # https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/ops.py#L12
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

@add_arg_scope
def conv(filter_size,
         stride,
         out_dim,
         layer_dict,
         bn=False,
         nl=tf.identity,
         init_w=None,
         init_b=tf.zeros_initializer(),
         padding='SAME',
         pad_type='ZERO',
         trainable=True,
         is_training=None,
         name='conv'):
    inputs = layer_dict['cur_input']
    stride = layers.get_shape4D(stride)
    in_dim = inputs.get_shape().as_list()[-1]
    filter_shape = layers.get_shape2D(filter_size) + [in_dim, out_dim]

    if padding == 'SAME' and pad_type == 'REFLECT':
        pad_size_1 = int((filter_shape[0] - 1) / 2)
        pad_size_2 = int((filter_shape[1] - 1) / 2)
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
            "REFLECT")
        padding = 'VALID'

    with tf.variable_scope(name):
        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)

        outputs = tf.nn.conv2d(inputs,
                               filter=weights,
                               strides=stride,
                               padding=padding,
                               use_cudnn_on_gpu=True,
                               data_format='NHWC',
                               dilations=[1, 1, 1, 1],
                               name='conv2d')
        outputs += biases

        if bn is True:
            outputs = layers.batch_norm(outputs, train=is_training, name='bn')

        layer_dict['cur_input'] = nl(outputs)
        return layer_dict['cur_input']

def unpool_2d(pool, 
              ind, 
              stride=[1, 2, 2, 1], 
              scope='unpool_2d'):
  """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """

  with tf.variable_scope(scope):
    ind_shape = tf.shape(ind)
    # pool = pool[:, :ind_shape[1], :ind_shape[2], :]

    input_shape = tf.shape(pool)
    output_shape = [input_shape[0],
                    input_shape[1] * stride[1],
                    input_shape[2] * stride[2],
                    input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0],
                         output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(
        tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
        shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0],
                        set_input_shape[1] * stride[1],
                        set_input_shape[2] * stride[2],
                        set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret

def max_pool(x,
             name='max_pool',
             filter_size=2,
             stride=None,
             padding='VALID',
             switch=False):
    """ 
    Max pooling layer 
    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 2): size of filter
        stride (int or list with length 2): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.
    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)

    if switch == True:
        return tf.nn.max_pool_with_argmax(
            x,
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            Targmax=tf.int64,
            name=name)
    else:
        return tf.nn.max_pool(
            x,
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            name=name), None

@add_arg_scope
def transpose_conv(x,
                   filter_size,
                   out_dim,
                   data_dict=None,
                   out_shape=None,
                   use_bias=True,
                   reuse=False,
                   stride=2,
                   padding='SAME',
                   trainable=True,
                   nl=tf.identity,
                   name='dconv'):

    stride = get_shape4D(stride)

    in_dim = x.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    x_shape = tf.shape(x)
    # assume output shape is input_shape*stride
    if out_shape is None:
        out_shape = tf.stack([x_shape[0],
                              tf.multiply(x_shape[1], stride[1]), 
                              tf.multiply(x_shape[2], stride[2]),
                              out_dim])        

    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]

    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
            init_w = None
            init_b = None
        elif data_dict is None:
            init_w = None
            init_b = None
        else:
            try:
                load_data = data_dict[name][0]
            except KeyError:
                load_data = data_dict[name]['weights']
            print('Load {} weights!'.format(name))
            # load_data = np.reshape(load_data, shape)
            # load_data = tf.nn.l2_normalize(
            #     tf.transpose(load_data, perm=[1, 0, 2, 3]))
            # load_data = tf.transpose(load_data, perm=[1, 0, 2, 3])
            init_w = tf.constant_initializer(load_data)

            if use_bias:
                try:
                    load_data = data_dict[name][1]
                except KeyError:
                    load_data = data_dict[name]['biases']
                print('Load {} biases!'.format(name))
                init_b = tf.constant_initializer(load_data)

        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)
        if use_bias:
            biases = tf.get_variable('biases',
                                     [in_dim],
                                     initializer=init_b,
                                     trainable=trainable)
            x = tf.nn.bias_add(x, biases)

        output = tf.nn.conv2d_transpose(x,
                                        weights, 
                                        output_shape=out_shape, 
                                        strides=stride, 
                                        padding=padding, 
                                        name=scope.name)

        # if use_bias:
        #     output = tf.nn.bias_add(output, biases)
        # TODO need test
        output.set_shape([None, None, None, out_dim])

        output = nl(output, name='output')
        return output
