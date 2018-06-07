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

sys.path.append('../')
from lib.dataflow.mnist import MNISTData 
from lib.model.ram import RAMClassification
from lib.helper.trainer import Trainer

if platform.node() == 'Qians-MacBook-Pro.local':
    DATA_PATH = '/Users/gq/Google Drive/Foram/CNN Data/code/GAN/MNIST_data/'
elif platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/MNIST_data/'
else:
    DATA_PATH = 'E://Dataset//MNIST//'

save_path = 'E:/tmp/tmp/'

BATCH_SIZE = 64
N_STEP = 6
N_SAMPLE = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true',
                        help='Run prediction')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')

    return parser.parse_args()

if __name__ == '__main__':
    FLAGS = get_args()

    train_data = MNISTData('train', data_dir=DATA_PATH, shuffle=True)
    train_data.setup(epoch_val=0, batch_size=BATCH_SIZE)
    # print(train_data.next_batch_dict())
    valid_data = MNISTData('val', data_dir=DATA_PATH, shuffle=True)
    valid_data.setup(epoch_val=0, batch_size=10)
    batch_data = valid_data.next_batch_dict()

    model = RAMClassification(im_size=28,
                              im_channel=1,
                              glimpse_scale=8,
                              n_loc_sample=N_STEP,
                              time_step=N_STEP,
                              n_class=10,
                              max_grad_norm=5.0,
                              loc_std=0.11)
    model.create_model()

    trainer = Trainer(model, train_data)
    writer = tf.summary.FileWriter(save_path)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.train:
            for step in range(0, 100):
                trainer.train_epoch(sess, N_SAMPLE)
                trainer.valid_epoch(sess, valid_data, BATCH_SIZE)
                saver.save(sess, '{}ram-center-mnist'.format(save_path), global_step=step)
        if FLAGS.predict:
            saver.restore(sess, '{}ram-center-mnist-99'.format(save_path))
            # writer.add_graph(sess.graph)
            trainer.test_batch(sess, batch_data, size=8.0)
        

        writer.close()
