#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: predictor.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Predictor(object):
    def __init__(self, model):
        self._model = model

        self._accuracy_op = model.get_accuracy()
        self._pred_op = model.layers['pred']
        self._sample_loc_op = model.layers['loc_sample']

    def evaluate(self, sess, dataflow, batch_size=None):
        self._model.set_is_training(False)

        step = 0
        acc_sum = 0
        while dataflow.epochs_completed == 0:
            step += 1
            batch_data = dataflow.next_batch_dict()
            acc = sess.run(
                self._accuracy_op, 
                feed_dict={self._model.image: batch_data['data'],
                           self._model.label: batch_data['label'],
                           })
            acc_sum += acc
        print('accuracy: {:.4f}'
              .format(acc_sum * 1.0 / step))

        self._model.set_is_training(True)

    def test_batch(self, sess, batch_data, unit_pixel, size, scale, save_path=''):
        def draw_bbx(ax, x, y):
            rect = patches.Rectangle(
                (x, y), cur_size, cur_size, edgecolor='r', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        self._model.set_is_training(False)
        
        test_im = batch_data['data']
        loc_list, pred, input_im, glimpses = sess.run(
            [self._sample_loc_op, self._pred_op, self._model.input_im,
             self._model.layers['retina_reprsent']],
            feed_dict={self._model.image: test_im,
                       self._model.label: batch_data['label'],
                        })

        pad_r = size * (2 ** (scale - 2))
        print(pad_r)
        im_size = input_im[0].shape[0]
        loc_list = np.clip(np.array(loc_list), -1.0, 1.0)
        loc_list = loc_list * 1.0 * unit_pixel / (im_size / 2 + pad_r)
        loc_list = (loc_list + 1.0) * 1.0 / 2 * (im_size + pad_r * 2)
        offset = pad_r

        print(pred)
        for step_id, cur_loc in enumerate(loc_list):
            im_id = 0
            glimpse = glimpses[step_id]
            for im, loc, cur_glimpse in zip(input_im, cur_loc, glimpse):
                im_id += 1                
                fig, ax = plt.subplots(1)
                ax.imshow(np.squeeze(im), cmap='gray')
                for scale_id in range(0, scale):
                    cur_size = size * 2 ** scale_id
                    side = cur_size * 1.0 / 2
                    x = loc[1] - side - offset
                    y = loc[0] - side - offset
                    draw_bbx(ax, x, y)
                # plt.show()
                for i in range(0, scale):
                    scipy.misc.imsave(
                        os.path.join(save_path,'im_{}_glimpse_{}_step_{}.png').format(im_id, i, step_id),
                        np.squeeze(cur_glimpse[:,:,i]))
                plt.savefig(os.path.join(
                    save_path,'im_{}_step_{}.png').format(im_id, step_id))
                plt.close(fig)

        self._model.set_is_training(True)
