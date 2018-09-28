#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Trainer(object):
    def __init__(self, model, train_data, init_lr=1e-3):
        self._model = model
        self._train_data = train_data
        self._lr = init_lr

        self._train_op = model.get_train_op()
        self._loss_op = model.get_loss()
        self._accuracy_op = model.get_accuracy()
        self._sample_loc_op = model.layers['loc_sample']
        self._pred_op = model.layers['pred']
        self._lr_op = model.cur_lr

        self.global_step = 0

    def train_epoch(self, sess, summary_writer=None):
        self._model.set_is_training(True)
        cur_epoch = self._train_data.epochs_completed
        step = 0
        loss_sum = 0
        acc_sum = 0
        while cur_epoch == self._train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = self._train_data.next_batch_dict()
            im = batch_data['data']
            label = batch_data['label']
            _, loss, acc, cur_lr = sess.run(
                [self._train_op, self._loss_op, self._accuracy_op, self._lr_op], 
                feed_dict={self._model.image: im,
                           self._model.label: label,
                           self._model.lr: self._lr})

            loss_sum += loss
            acc_sum += acc

            if step % 100 == 0:
                print('step: {}, loss: {:.4f}, accuracy: {:.4f}'
                      .format(self.global_step,
                              loss_sum * 1.0 / step,
                              acc_sum * 1.0 / step))

        print('epoch: {}, loss: {:.4f}, accuracy: {:.4f}, lr:{}'
              .format(cur_epoch,
                      loss_sum * 1.0 / step,
                      acc_sum * 1.0 / step, cur_lr))
        if summary_writer is not None:
            s = tf.Summary()
            s.value.add(tag='train/loss', simple_value=loss_sum * 1.0 / step)
            s.value.add(tag='train/accuracy', simple_value=acc_sum * 1.0 / step)
            summary_writer.add_summary(s, self.global_step)

    def valid_epoch(self, sess, dataflow, batch_size, summary_writer=None):
        self._model.set_is_training(False)
        dataflow.setup(epoch_val=0, batch_size=batch_size)

        step = 0
        loss_sum = 0
        acc_sum = 0
        while dataflow.epochs_completed == 0:
            step += 1
            batch_data = dataflow.next_batch_dict()
            loss, acc = sess.run(
                [self._loss_op, self._accuracy_op], 
                feed_dict={self._model.image: batch_data['data'],
                           self._model.label: batch_data['label'],
                           })
            loss_sum += loss
            acc_sum += acc
        print('valid loss: {:.4f}, accuracy: {:.4f}'
              .format(loss_sum * 1.0 / step, acc_sum * 1.0 / step))

        if summary_writer is not None:
            s = tf.Summary()
            s.value.add(tag='valid/loss', simple_value=loss_sum * 1.0 / step)
            s.value.add(tag='valid/accuracy', simple_value=acc_sum * 1.0 / step)
            summary_writer.add_summary(s, self.global_step)

        self._model.set_is_training(True)
