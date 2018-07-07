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
        self._sum_op = model.get_summary()

        self._lr_op = model.cur_lr

        self.global_step = 0

    def train_epoch(self, sess, summary_writer=None):
        self._model.set_is_training(True)
        # self._lr = np.maximum(self._lr * 0.97, 1e-4)
        self._lr = self._lr * 0.97
        cur_epoch = self._train_data.epochs_completed

        step = 0
        loss_sum = 0
        acc_sum = 0
        while cur_epoch == self._train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = self._train_data.next_batch_dict()

            _, loss, acc, cur_lr, cur_summary = sess.run(
                [self._train_op, self._loss_op, self._accuracy_op, self._lr_op, self._sum_op], 
                feed_dict={self._model.image: batch_data['im'],
                           self._model.cls_label: batch_data['label'],
                           self._model.lr: self._lr})

            loss_sum += loss
            acc_sum += acc

            if step % 100 == 0:
                print('step: {}, loss: {:.4f}, accuracy: {:.4f}'
                      .format(self.global_step,
                              loss_sum * 1.0 / step,
                              acc_sum * 1.0 / step))
                if summary_writer is not None:
                    s = tf.Summary()
                    s.value.add(tag='train_loss', simple_value=loss_sum * 1.0 / step)
                    s.value.add(tag='train_accuracy', simple_value=acc_sum * 1.0 / step)
                    summary_writer.add_summary(s, self.global_step)
                    summary_writer.add_summary(cur_summary, self.global_step)

        print('epoch: {}, loss: {:.4f}, accuracy: {:.4f}, lr:{}'
              .format(cur_epoch,
                      loss_sum * 1.0 / step,
                      acc_sum * 1.0 / step, cur_lr))
        

    def valid_epoch(self, sess, dataflow, batch_size):
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
                feed_dict={self._model.image: batch_data['im'],
                           self._model.cls_label: batch_data['label']})

            loss_sum += loss
            acc_sum += acc
        print('valid loss: {:.4f}, accuracy: {:.4f}'
              .format(loss_sum * 1.0 / step, acc_sum * 1.0 / step))

        self._model.set_is_training(True)

    # def test_batch(self, sess, batch_data, unit_pixel, size, scale, save_path=''):
    #     def draw_bbx(ax, x, y):
    #         rect = patches.Rectangle(
    #             (x, y), cur_size, cur_size, edgecolor='r', facecolor='none', linewidth=2)
    #         ax.add_patch(rect)

    #     self._model.set_is_training(False)
        
    #     test_im = batch_data['data']
    #     loc_list, pred, input_im, glimpses = sess.run(
    #         [self._sample_loc_op, self._pred_op, self._model.input_im,
    #          self._model.layers['retina_reprsent']],
    #         feed_dict={self._model.image: test_im,
    #                    self._model.label: batch_data['label'],
    #                     })

    #     pad_r = size * (2 ** (scale - 2))
    #     print(pad_r)
    #     im_size = input_im[0].shape[0]
    #     loc_list = np.clip(np.array(loc_list), -1.0, 1.0)
    #     loc_list = loc_list * 1.0 * unit_pixel / (im_size / 2 + pad_r)
    #     loc_list = (loc_list + 1.0) * 1.0 / 2 * (im_size + pad_r * 2)
    #     offset = pad_r

    #     print(pred)
    #     for step_id, cur_loc in enumerate(loc_list):
    #         im_id = 0
    #         glimpse = glimpses[step_id]
    #         for im, loc, cur_glimpse in zip(input_im, cur_loc, glimpse):
    #             im_id += 1                
    #             fig, ax = plt.subplots(1)
    #             ax.imshow(np.squeeze(im), cmap='gray')
    #             for scale_id in range(0, scale):
    #                 cur_size = size * 2 ** scale_id
    #                 side = cur_size * 1.0 / 2
    #                 x = loc[1] - side - offset
    #                 y = loc[0] - side - offset
    #                 draw_bbx(ax, x, y)
    #             # plt.show()
    #             for i in range(0, scale):
    #                 scipy.misc.imsave(
    #                     os.path.join(save_path,'im_{}_glimpse_{}_step_{}.png').format(im_id, i, step_id),
    #                     np.squeeze(cur_glimpse[:,:,i]))
    #             plt.savefig(os.path.join(
    #                 save_path,'im_{}_step_{}.png').format(im_id, step_id))
    #             plt.close(fig)

    #     self._model.set_is_training(True)