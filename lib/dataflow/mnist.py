#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import gzip
import struct
from datetime import datetime
import numpy as np 
# from tensorcv.dataflow.base import RNGDataFlow

_RNG_SEED = None

def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.

    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


def identity(im):
    return im

class MNISTData(object):
    """ class for MNIST dataflow

        To access the data of mini-batch, first get data of all the channels
        through batch_data = MNISTData.next_batch_dict()
        then use corresponding key to get label or image through
        batch_data[key].
    """
    def __init__(self, name, data_dir='', n_use_label=None, n_use_sample=None,
                 batch_dict_name=None, shuffle=True, pf=identity):
        """
        Args:
            name (str): name of data to be read (['train', 'test', 'val'])
            data_dir (str): directory of MNIST data
            n_use_label (int): number of labels to be used
            n_use_sample (int): number of samples to be used
            batch_dict_name (list of str): list of keys for 
                image and label of batch data
            shuffle (bool): whether shuffle data or not
            pf: pre-process function for image data
        """
        assert os.path.isdir(data_dir)
        self._data_dir = data_dir

        self._shuffle = shuffle
        self._pf = pf

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        assert name in ['train', 'test', 'val']
        self.setup(epoch_val=0, batch_size=1)

        self._load_files(name, n_use_label, n_use_sample)
        self._image_id = 0

    def next_batch_dict(self):
        batch_data = self.next_batch()
        data_dict = {key: data for key, data
                     in zip(self._batch_dict_name, batch_data)}
        return data_dict

    def _load_files(self, name, n_use_label, n_use_sample):
        if name == 'train':
            image_name = 'train-images-idx3-ubyte.gz'
            label_name = 'train-labels-idx1-ubyte.gz'
        else:
            image_name = 't10k-images-idx3-ubyte.gz'
            label_name = 't10k-labels-idx1-ubyte.gz'

        image_path = os.path.join(self._data_dir, image_name)
        label_path = os.path.join(self._data_dir, label_name)

        with gzip.open(label_path) as f:
            magic = struct.unpack('>I', f.read(4))
            if magic[0] != 2049:
                raise Exception('Invalid file: unexpected magic number.')
            n_label = struct.unpack('>I', f.read(4))
            label_list = np.fromstring(f.read(n_label[0]), dtype = np.uint8)

        with gzip.open(image_path) as f:
            magic = struct.unpack('>I', f.read(4))
            if magic[0] != 2051:
                raise Exception('Invalid file: unexpected magic number.')
            n_im, rows, cols = struct.unpack('>III', f.read(12))
            image_list = np.fromstring(
                f.read(n_im * rows * cols), dtype = np.uint8)
            image_list = np.reshape(image_list, (n_im, rows, cols, 1))
            im_list = []
            if n_use_sample is not None and n_use_sample < len(label_list):
                remain_sample = n_use_sample // 10 * 10
                left_sample = n_use_sample - remain_sample
                keep_sign = [0 for i in range(10)]
                data_idx = 0
                new_label_list = []
                for idx, im in enumerate(image_list):

                    if remain_sample > 0:
                        if keep_sign[label_list[idx]] < (n_use_sample // 10):
                            keep_sign[label_list[idx]] += 1
                            im_list.append(self._pf(im))
                            new_label_list.append(label_list[idx])
                            remain_sample -= 1
                    else:
                        break
                im_list.extend(image_list[idx:idx + left_sample])
                new_label_list.extend(label_list[idx:idx + left_sample])
                label_list = new_label_list

            else:
                for im in image_list:
                    im_list.append(self._pf(im))

        self.im_list = np.array(im_list)
        self.label_list = np.array(label_list)

        if n_use_label is not None and n_use_label < self.size():
            remain_sample = n_use_label // 10 * 10
            left_sample = n_use_label - remain_sample
            keep_sign = [0 for i in range(10)]
            data_idx = 0
            while remain_sample > 0:
                if keep_sign[self.label_list[data_idx]] < (n_use_label // 10):
                    keep_sign[self.label_list[data_idx]] += 1
                    remain_sample -= 1
                else:
                    self.label_list[data_idx] = 10
                data_idx += 1

            self.label_list[data_idx + left_sample:] = 10
        self._suffle_files()

    def _suffle_files(self):
        if self._shuffle:
            idxs = np.arange(self.size())

            self.rng.shuffle(idxs)
            self.im_list = self.im_list[idxs]
            self.label_list = self.label_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_files = self.im_list[start:end]
        batch_label = self.label_list[start:end]

        if self._image_id + self._batch_size > self.size():
            self._epochs_completed += 1
            self._image_id = 0
            self._suffle_files()
        return [batch_files, batch_label]

    def setup(self, epoch_val, batch_size, **kwargs):
        self._epochs_completed = epoch_val
        self._batch_size = batch_size
        self.rng = get_rng(self)
        try:
            self._suffle_files()
        except AttributeError:
            pass

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # File: mnist.py
# # Author: Qian Ge <geqian1001@gmail.com>

# import os
# import numpy as np 
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorcv.dataflow.base import RNGDataFlow

# def identity(im):
#     return im

# def concat_pair(im_1, im_2):
#     im_1 = np.expand_dims(im_1, axis=-1)
#     im_2 = np.expand_dims(im_2, axis=-1)
#     return np.concatenate((im_1, im_2), axis=-1)

# def get_mnist_im_label(name, mnist_data):
#     if name == 'train':
#         return mnist_data.train.images, mnist_data.train.labels
#     elif name == 'val':
#         return mnist_data.validation.images, mnist_data.validation.labels
#     else:
#         return mnist_data.test.images, mnist_data.test.labels

# class MNISTData(RNGDataFlow):
#     def __init__(self, name, batch_dict_name=None, data_dir='', shuffle=True, pf=identity):
#         assert os.path.isdir(data_dir)
#         self._data_dir = data_dir

#         self._shuffle = shuffle
#         if pf is None:
#             pf = identity
#         self._pf = pf

#         if not isinstance(batch_dict_name, list):
#             batch_dict_name = [batch_dict_name]
#         self._batch_dict_name = batch_dict_name

#         assert name in ['train', 'test', 'val']
#         self.setup(epoch_val=0, batch_size=1)

#         self._load_files(name)
#         self._image_id = 0

#     def next_batch_dict(self):
#         batch_data = self.next_batch()
#         data_dict = {key: data for key, data in zip(self._batch_dict_name, batch_data)}
#         return data_dict

#     def _load_files(self, name):
#         mnist_data = input_data.read_data_sets(self._data_dir, one_hot=False)
#         self.im_list = []
#         self.label_list = []

#         mnist_images, mnist_labels = get_mnist_im_label(name, mnist_data)
#         for image, label in zip(mnist_images, mnist_labels):
#             # TODO to be modified
#             image = np.reshape(image, [28, 28, 1])
            
#             # image = np.reshape(image, [28, 28, 1])
            
#             self.im_list.append(image)
#             self.label_list.append(label)
#         self.im_list = np.array(self.im_list)
#         self.label_list = np.array(self.label_list)

#         self._suffle_files()

#     def _suffle_files(self):
#         if self._shuffle:
#             idxs = np.arange(self.im_list.shape[0])

#             self.rng.shuffle(idxs)
#             self.im_list = self.im_list[idxs]
#             self.label_list = self.label_list[idxs]

#     def size(self):
#         return self.im_list.shape[0]

#     def next_batch(self):
#         assert self._batch_size <= self.size(), \
#           "batch_size {} cannot be larger than data size {}".\
#            format(self._batch_size, self.size())
#         start = self._image_id
#         self._image_id += self._batch_size
#         end = self._image_id
#         batch_files = []
#         for im in self.im_list[start:end]:
#             im = np.reshape(im, [28, 28])
#             im = self._pf(im)
#             im = np.expand_dims(im, axis=-1)
#             batch_files.append(im)

#         batch_label = self.label_list[start:end]

#         if self._image_id + self._batch_size > self.size():
#             self._epochs_completed += 1
#             self._image_id = 0
#             self._suffle_files()
#         return [batch_files, batch_label]


# class MNISTPair(MNISTData):
#     def __init__(self,
#                  name,
#                  label_dict,
#                  batch_dict_name=None,
#                  data_dir='',
#                  shuffle=True,
#                  pf=identity,
#                  pairprocess=concat_pair,
#                  ):
#         self._pair_fnc = pairprocess
#         self._label_dict = label_dict
#         super(MNISTPair, self).__init__(name=name,
#                                         batch_dict_name=batch_dict_name,
#                                         data_dir=data_dir,
#                                         shuffle=shuffle,
#                                         pf=pf,)

#     def size(self):
#         return int(np.floor(self.im_list.shape[0] / 2.0))

#     def next_batch(self):
#         assert self._batch_size <= self.size(), \
#           "batch_size {} cannot be larger than data size {}".\
#            format(self._batch_size, self.size())
#         # start = self._image_id
#         # self._image_id += self._batch_size * 2
#         # end = self._image_id
#         batch_files = []
#         batch_label = []
#         start = self._image_id
#         for data_id in range(0, self._batch_size):
#             im_1 = np.reshape(self.im_list[start], [28, 28])
#             im_2 = np.reshape(self.im_list[start + 1], [28, 28])
#             im = self._pair_fnc(im_1, im_2)
#             im = np.expand_dims(im, axis=-1)
#             batch_files.append(im)

#             label_1 = self.label_list[start]
#             label_2 = self.label_list[start + 1]
#             label = self._label_dict['{}{}'.format(label_1, label_2)]
#             batch_label.append(label)
#             start = start + 2
#         end = start
#         self._image_id = end

#         if self._image_id + self._batch_size > self.size():
#             self._epochs_completed += 1
#             self._image_id = 0
#             self._suffle_files()
#         return [batch_files, batch_label]
