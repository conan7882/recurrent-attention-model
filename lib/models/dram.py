#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dram.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import tensorcv.models.layers as layers
from tensorcv.models.base import BaseModel
import lib.models.layers as L
from lib.models.distribution import sample_normal_single


INIT_W = tf.keras.initializers.he_normal()
# INIT_W = tf.random_normal_initializer(stddev=0.02)

class DRAM(BaseModel):
    def __init__(self,
                 im_size,
                 n_channel,
                 n_hidden,
                 n_linear_hidden,
                 n_step,
                 location_std,
                 glimpse_base_size,
                 n_glimpse_scale,
                 unit_pixel,
                 n_class=10,
                 coarse_size=64,
                 loc_weight=1e-3):
        self._n_class = n_class
        self._n_channel = n_channel
        self._im_size = layers.get_shape2D(im_size)
        self._coarse_size = layers.get_shape2D(coarse_size)
        if not isinstance(n_hidden, list):
            n_hidden = [n_hidden]
        assert len(n_hidden) == 2
        self._n_hidden = n_hidden
        self._n_linear_hidden = n_linear_hidden
        self._n_step = n_step
        self._l_std = location_std
        self._g_size = glimpse_base_size
        self._g_n = n_glimpse_scale
        self._lw = loc_weight

        # assume square images
        # self._unit_pixel = unit_pixel
        self._l_range = self._im_size[0] / 2.0 / (1.0 * unit_pixel)

        self.layers = {}
        self.set_is_training(True)
        
    def create_model(self):
        
        self._create_input()
        with tf.variable_scope('core_net', reuse=tf.AUTO_REUSE):
            self._core_net(self.image)

    def _create_input(self):
        self.lr = tf.placeholder(tf.float32, name="lr")
        # self.label = tf.placeholder(tf.float32, 
        #                             [None, 4],
        #                             name="label")
        self.image = tf.placeholder(tf.float32,
                                    [None, None, None, self._n_channel],
                                    name="image")
        self.label = tf.placeholder(tf.int64, [None], name="label")

    def create_test_model(self):
        self.set_is_training(False)
        self._create_test_input()
        with tf.variable_scope('core_net', reuse=tf.AUTO_REUSE):
            self._core_net(self.test_image)

    def _create_test_input(self):
        self.label = tf.placeholder(tf.int64, [None], name="label")
        self.test_image = tf.placeholder(tf.float32,
                                         [None, None, None, self._n_channel],
                                         name="test_image")

    def _core_net(self, inputs):
        self.layers['retina_reprsent'] = []
        self.layers['loc_sample'] = []
        self.layers['loc_mean'] = []
        self.layers['reward'] = []
        self.layers['rnn_outputs'] = []
        def _make_cell(hidden_size):
            return L.make_cell(hidden_size,
                               forget_bias=1.0,
                               is_training=self.is_training,
                               keep_prob=1.0)

        with tf.name_scope('init'):
            rnn_bottom = _make_cell(self._n_hidden[0])
            rnn_top = _make_cell(self._n_hidden[1])

            context = self._context_net(inputs)
            rnn_top_state = context
            b_size = tf.shape(inputs)[0]
            rnn_bottom_state = rnn_bottom.zero_state(b_size, tf.float32)

        for step_id in range(0, self._n_step):
            with tf.variable_scope('step', reuse=tf.AUTO_REUSE):
                if step_id == 0:
                    init_input_layer_2 = tf.zeros((b_size, self._n_hidden[0]), tf.float32)
                    with tf.variable_scope('rnn_top'):
                        rnn_out, rnn_top_state = rnn_top(init_input_layer_2, rnn_top_state)
                else:
                    with tf.variable_scope('rnn_bottom'):
                        rnn_bottom_out, rnn_bottom_state = rnn_bottom(rnn_inputs, rnn_bottom_state)
                    with tf.variable_scope('rnn_top'):
                        rnn_out, rnn_top_state = rnn_top(rnn_bottom_out, rnn_top_state)

                if step_id < self._n_step - 1:
                    self.layers['rnn_outputs'].append(rnn_out)
                    l_mean, l_sample = self._emission_net(rnn_out)
                    if self.is_training == False:
                        l_sample = l_mean
                    self.layers['loc_sample'].append(l_sample)
                    self.layers['loc_mean'].append(l_mean)
                    glimpse_out = self._glimpse_net(l_sample)
                    self.g = glimpse_out
                    rnn_inputs = glimpse_out

        # self._action_net(rnn_final_out=rnn_out)
        self.layers['cls_logits'] = self._action_cls_net(rnn_final_out=rnn_bottom_out)
        self.layers['pred'] = tf.argmax(self.layers['cls_logits'], axis=1) 

    def _context_net(self, inputs):
        with tf.variable_scope("context"):
            self.layers['cur_input'] = tf.image.resize_images(
                inputs,
                size=self._coarse_size,
                method=tf.image.ResizeMethod.BILINEAR,
                align_corners=False
                )
            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.conv],
                            layer_dict=self.layers,
                            is_training=self.is_training,
                            bn=False,
                            stride=2,
                            nl=tf.nn.relu,
                            init_w=INIT_W,
                            ):

                L.conv(filter_size=5, out_dim=16, name='conv1')
                L.conv(filter_size=3, out_dim=32, name='conv2')
                L.conv(filter_size=3, out_dim=64, name='conv3')

            out_dim = self._n_hidden[-1] * 2
            L.linear(out_dim=out_dim,
                     layer_dict=self.layers,
                     name='linear',
                     # nl=tf.nn.relu,
                     # init_w=INIT_W
                     )
            context = self.layers['cur_input']
            outputs = tf.contrib.rnn.LSTMStateTuple(c=context[:, :self._n_hidden[-1]],
                                                    h=context[:, self._n_hidden[-1]:])

        return outputs

    def _emission_net(self, inputs):
        # self.layers['cur_input'] = tf.stop_gradient(inputs)
        self.layers['cur_input'] = inputs
        with tf.variable_scope("emission"):
            l_mean = L.linear(out_dim=2,
                              layer_dict=self.layers,
                              name='l_mean',
                              nl=tf.identity,
                              # init_w=INIT_W
                              )
            l_mean = tf.clip_by_value(l_mean, -self._l_range, self._l_range)
            l_mean_sample = sample_normal_single(l_mean, stddev=self._l_std)
            l_mean_sample = tf.clip_by_value(l_mean_sample, -self._l_range, self._l_range)
            return l_mean, tf.stop_gradient(l_mean_sample)

    def _glimpse_net(self, l_sample):
        """
            Args:
                inputs: [batch, h, w, c]
                l_sample: [batch, 2]
        """
        inputs = self.image
        with tf.name_scope("glimpse_sensor"):
            max_r = int(self._g_size * (2 ** (self._g_n - 2)))
            inputs_pad = tf.pad(
                inputs,
                [[0, 0], [max_r, max_r], [max_r, max_r], [0, 0]],
                'CONSTANT') 
            l_sample_normal = l_sample * 1.0 / self._l_range
            l_sample_adj = l_sample_normal * 1.0 * (self._im_size[0] / 2) / (self._im_size[0] / 2. + max_r)
            # self.test = l_sample_adj
            
            retina_reprsent = []
            for g_id in range(0, self._g_n):
                cur_size = self._g_size * (2 ** g_id)
                cur_glimpse = tf.image.extract_glimpse(
                    inputs_pad,
                    size=[cur_size, cur_size],
                    offsets=l_sample_adj,
                    centered=True,
                    normalized=True,
                    uniform_noise=True,
                    name='glimpse_sensor',
                    )
                cur_glimpse = tf.image.resize_images(
                    cur_glimpse,
                    size=[self._g_size, self._g_size],
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    )
                retina_reprsent.append(cur_glimpse)
            retina_reprsent = tf.concat(retina_reprsent, axis=-1)
            self.layers['retina_reprsent'].append(retina_reprsent)
            
        with tf.variable_scope("glimpse_net"):
            out_dim = self._n_linear_hidden
            with tf.variable_scope("G_image"):
                self.layers['cur_input'] = retina_reprsent
                arg_scope = tf.contrib.framework.arg_scope
                with arg_scope([L.conv],
                                layer_dict=self.layers,
                                is_training=self.is_training,
                                bn=True,
                                stride=1,
                                nl=tf.nn.relu,
                                init_w=INIT_W,
                                ):
                    L.conv(filter_size=5, out_dim=32, name='conv1')
                    L.conv(filter_size=3, out_dim=64, name='conv2')
                    L.conv(filter_size=3, out_dim=128, name='conv3')
                g_image = L.linear(out_dim=out_dim,
                                   layer_dict=self.layers,
                                   name='linear',
                                   nl=tf.nn.relu,
                                   # init_w=INIT_W
                                   )
            with tf.variable_scope("G_location"):
                g_loc = L.linear(inputs=l_sample,
                                 out_dim=out_dim,
                                 layer_dict=self.layers,
                                 name='g_loc',
                                 nl=tf.nn.relu,
                                 # init_w=INIT_W
                                 )
            g = tf.concat((g_image, g_loc), axis=-1)
            g = tf.multiply(g_image, g_loc, "what_where")
            return g

    def _action_cls_net(self, rnn_final_out):
        with tf.variable_scope('action_cls'):
            act = L.linear(
                inputs=rnn_final_out,
                layer_dict=self.layers,
                out_dim=self._n_class,
                # init_w=INIT_W,
                name='act_cls')
            return act

    def _REINFORCE(self):
        with tf.name_scope('REINFORCE'):
            labels = self.label
            pred = self.layers['pred']
            reward = tf.stop_gradient(tf.cast(tf.equal(pred, labels), tf.float32))
            self.reward = reward
            reward = tf.tile(tf.expand_dims(reward, 1), [1, self._n_step - 1]) # [b_size, n_step]

            loc_mean = tf.stack(self.layers['loc_mean']) # [n_step, b_size, 2]
            loc_sample = tf.stack(self.layers['loc_sample']) # [n_step, b_size, 2]
            dist = tf.distributions.Normal(loc=loc_mean, scale=self._l_std)
            log_prob = dist.log_prob(loc_sample) # [n_step, b_size, 2]
            log_prob = tf.reduce_sum(log_prob, -1) # [n_step, b_size]
            log_prob = tf.transpose(log_prob) # [b_size, n_step]

            baselines = self._comp_baselines()
            b_mse = tf.losses.mean_squared_error(labels=reward,
                                                 predictions=baselines)
            low_var_reward = (reward - tf.stop_gradient(baselines))
            REINFORCE_reward = tf.reduce_mean(log_prob * low_var_reward)

            loss = -REINFORCE_reward + b_mse
            return loss

    def _comp_baselines(self):
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
            # core net does not trained through baseline loss
            rnn_outputs = tf.stop_gradient(self.layers['rnn_outputs'])
            baselines = []
            for step_id in range(0, self._n_step-1):
                b = L.linear(
                    inputs=rnn_outputs[step_id],
                    layer_dict=self.layers,
                    out_dim=1,
                    # init_w=INIT_W,
                    name='baseline')
                b = tf.squeeze(b, axis=-1)
                baselines.append(b)
            
            baselines = tf.stack(baselines) # [n_step, b_size]
            baselines = tf.transpose(baselines) # [b_size, n_step]
            return baselines

    def _cls_loss(self):
        with tf.name_scope('class_cross_entropy'):
            labels = self.label
            # if self.is_training:
                # label = tf.tile(label, [self._n_l_sample])
            logits = self.layers['cls_logits']
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            cross_entropy = tf.reduce_mean(cross_entropy)
            return cross_entropy

    def get_loss(self):
        try:
            return self.loss
        except AttributeError:
            self.loss = self._get_loss()
            return self.loss

    def _get_loss(self):
        self.REINFORCE_loss = self._REINFORCE()
        self.cls_loss = self._cls_loss()
        # return self.REINFORCE_loss + self.cls_loss
        return self._lw * self.REINFORCE_loss + self.cls_loss

    def get_train_op(self):
        # global_step = tf.get_variable(
        #     'global_step',
        #     [],
        #     initializer=tf.constant_initializer(0),
        #     trainable=False)
        # cur_lr = tf.train.exponential_decay(self.lr,
        #                                     global_step=global_step,
        #                                     decay_steps=40,
        #                                     # decay_steps=1,
        #                                     # decay_steps=1000,
        #                                     decay_rate=0.97,
        #                                     # decay_rate=0.99,
        #                                     staircase=True,
        #                                     )
        # cur_lr = tf.maximum(cur_lr, self.lr / 100.0)
        self.cur_lr = self.lr

        loss = self.get_loss()
        var_list = tf.trainable_variables()
        grads = tf.gradients(loss, var_list)
        [tf.summary.histogram('gradient/' + var.name, grad, 
         collections = [tf.GraphKeys.SUMMARIES])
         for grad, var in zip(grads, var_list)]
        grads, _ = tf.clip_by_global_norm(grads, 5)
        # opt = tf.train.AdamOptimizer(self.cur_lr)
        opt = tf.train.RMSPropOptimizer(learning_rate=self.cur_lr, momentum=0.9)
        train_op = opt.apply_gradients(zip(grads, var_list))
        # train_op = opt.apply_gradients(zip(grads, var_list),
        #                                global_step=global_step)
        return train_op

    def get_accuracy(self):
        labels = self.label
        pred = self.layers['pred']
        return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
        # return tf.reduce_mean(tf.cast(self.layers['reward'], tf.float32))

    def get_summary(self):
        # for i in range(0, self._n_step-1):
        #     tf.summary.image('glimpse_{}'.format(i),
        #                      tf.expand_dims(self.layers['retina_reprsent'][i][:,:,:,1], -1))
        # emission_vars = [v for v in tf.trainable_variables() 
        #                 if v.name.startswith('emission/')]
        return tf.summary.merge_all()
