#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist_center.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import numpy as np
import tensorflow as tf
import platform
import scipy.misc
import argparse
from read_mnist import trans_mnist

sys.path.append('../')
from lib.dataflow.mnist import MNISTData 
from lib.models.ram import RAMClassification
from lib.helper.trainer import Trainer

DATA_PATH = '/home/qge2/workspace/data/MNIST_data/'
SAVE_PATH = '/home/qge2/workspace/data/out/ram/'
RESULT_PATH = '/home/qge2/workspace/data/out/ram/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true',
                        help='Run prediction')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    parser.add_argument('--trans', action='store_true',
                        help='Transform image')
    parser.add_argument('--center', action='store_true',
                        help='Center')

    parser.add_argument('--step', type=int, default=1,
                        help='Number of glimpse')
    parser.add_argument('--sample', type=int, default=1,
                        help='Number of location samples during training')
    parser.add_argument('--glimpse', type=int, default=12,
                        help='Glimpse base size')
    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Max number of epoch')
    parser.add_argument('--load', type=int, default=100,
                        help='Load pretrained parameters with id')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Init learning rate')
    parser.add_argument('--std', type=float, default=0.11,
                        help='std of location')
    parser.add_argument('--pixel', type=int, default=26,
                        help='unit_pixel')
    parser.add_argument('--scale', type=int, default=3,
                        help='scale of glimpse')
    
    return parser.parse_args()

class config_center():
    step = 6
    sample = 1
    glimpse = 8
    n_scales = 1
    batch = 128
    epoch = 1000
    loc_std = 0.03
    unit_pixel = 12
    im_size = 28

class config_transform():
    step = 6
    sample = 1
    glimpse = 12
    n_scales = 3
    batch = 128
    epoch = 2000
    loc_std = 0.03
    unit_pixel = 26
    im_size = 60

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.trans:
        name = 'trans'
        config = config_transform()
    elif FLAGS.center:
        name = 'centered'
        config = config_center()
    else:
        FLAGS.trans = True
        name = 'custom'
        class config_FLAGS():
            step = FLAGS.step
            sample = FLAGS.sample
            glimpse = FLAGS.glimpse
            n_scales = FLAGS.scale
            batch = FLAGS.batch
            epoch = FLAGS.epoch
            loc_std = FLAGS.std
            unit_pixel = FLAGS.pixel
        config = config_FLAGS()

    train_data, valid_data = trans_mnist(trans_size=config.im_size, batch_size=config.batch)

    # train_data = MNISTData('train', data_dir=DATA_PATH, shuffle=True)
    # train_data.setup(epoch_val=0, batch_size=config.batch)
    # valid_data = MNISTData('val', data_dir=DATA_PATH, shuffle=True)
    # valid_data.setup(epoch_val=0, batch_size=10)

    model = RAMClassification(
                              im_size=config.im_size,
                              im_channel=1,
                              glimpse_base_size=config.glimpse,
                              n_glimpse_scale=config.n_scales,
                              n_loc_sample=config.sample,
                              n_step=config.step,
                              n_class=10,
                              max_grad_norm=5.0,
                              unit_pixel=config.unit_pixel,
                              loc_std=config.loc_std,
                              # is_transform=FLAGS.trans
                              )
    

    trainer = Trainer(model, train_data, init_lr=FLAGS.lr)
    writer = tf.summary.FileWriter(SAVE_PATH)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.train:
            writer.add_graph(sess.graph)
            for step in range(0, config.epoch):
                trainer.train_epoch(sess, summary_writer=writer)
                trainer.valid_epoch(sess, valid_data, config.batch)
                saver.save(sess, 
                           '{}ram-{}-mnist-step-{}'
                           .format(SAVE_PATH, name, config.step),
                           global_step=step)
                writer.close()

        if FLAGS.predict:
            valid_data.setup(epoch_val=0, batch_size=20)
            saver.restore(sess, 
                          '{}ram-{}-mnist-step-6-{}'
                          .format(SAVE_PATH, name, FLAGS.load))
            
            batch_data = valid_data.next_batch_dict()
            trainer.test_batch(
                sess,
                batch_data,
                unit_pixel=config.unit_pixel,
                size=config.glimpse,
                scale=config.n_scales,
                save_path=RESULT_PATH)
