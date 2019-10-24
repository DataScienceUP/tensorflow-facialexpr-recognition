import os
import glob

import numpy as np
import tensorflow as tf

from lib.config import DataConfig
from lib.config import ConfigReader


class DataLoader(object):
    def __init__(self, config, training_mode, shuffle):
        """
        :param config:
        :param training_mode:
        :param shuffle:
        """
        self.config = config
        self.training_mode = training_mode
        self.shuffle = shuffle


class _CNN_Data_Loader(DataLoader):
    def __init__(self, config, name, training_mode, shuffle):
        """
        This class and functions are the key for loading the data into the model graph in our training loop
        :param config:
        :param name:
        :param training_mode:
        :param shuffle:
        """
        super().__init__(config, training_mode, shuffle)
        self.data_dir = self.config.data_dir
        self.image_width = self.config.image_width
        self.image_height = self.config.image_height
        self.image_depth = self.config.image_depth
        self.data_dir = self.config.data_dir
        self.batch_size = self.config.batch_size
        self.n_classes = self.config.n_classes
        self.normalize = self.config.normalize
        self.name = name

    def _normalization(self, tensor):
        """
        This is a max min tensorflow operation, in general is not need it as you can scale the images img =/255 or use
        simple operations
        :param tensor: the batch of images
        :return:
        """
        tensor = tf.div(tf.subtract(tensor, tf.reduce_min(tensor)), tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))
        return tensor

    def _parse_record(self, tf_record):
        """
        :param tf_record: tf record selected by the tf.dataset api
        :return: img and label decoded
        """
        # this part is important: you have to take into account the encoding of the observations in the record
        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'crop': tf.FixedLenFeature([], tf.string)
        }
        record = tf.parse_single_example(tf_record, features)
        crop = tf.decode_raw(record['crop'], tf.uint8)
        height = tf.cast(record['height'], tf.int32)
        width = tf.cast(record['width'], tf.int32)
        depth = tf.cast(record['depth'], tf.int32)
        # tensorflow needs a reshaping to evaluate the inputs over the graph in this case we have 48,48,1 images
        image_raw = tf.reshape(crop, shape=(height, width, depth))
        #
        label = tf.cast(record['label'], tf.int32)
        # tensorflow build operations can handle the one hot encoding for you
        label = tf.one_hot(label, depth=self.n_classes)
        #
        # apply tensorflow operation for normalization
        if self.normalize:
            # image_raw = self._normalization(image_raw)
            image_raw /=255
        return image_raw, label

    def _generate_batch(self):
        """
        This function should generate a batch of data according to the batch size used ex 64, 128, 256 images
        """
        # first we define a file pattern in the case of training name will be 'train'
        data_filenames = '{}/{}_*.tfrecord'.format(self.data_dir, self.name)
        num_files = len(glob.glob(data_filenames))
        # we shuffle all filenames with the API in order to get observations from every record
        dataset = tf.data.Dataset.list_files(data_filenames).shuffle(7)
        dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename), cycle_length=num_files)
        #  if you like more classic python you may try
        #  data_filenames = glob.glob('{}/test_*.tfrecord'.format(self.data_dir)) however in some cases you have to be
        # careful because you may load consecutive information from one single file, to solve that modify also
        # the next statements

        # we map the decoder
        dataset = dataset.map(self._parse_record)
        # we shuffle the data according to a buffer size (observations that we can take in memory), in several cases
        # you may have millons of observations so using an input pipeline is important (here and in pytroch for example)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=3000 + 3 * self.batch_size)
        # I did not train in terms of epochs ( iterations / batch size) = epochs
        # we use instead unlimited repetitions of the data so we work with all global iterations that we want,
        #  if the dim of the dataset is around 28900 training crops and batches of 64, then 28900/64 = 451
        #  iterations to finish 1 epoch
        dataset = dataset.repeat()
        # ending an epoch means  32000/128 is not exact so we have some remaning examples inconsistent with tha batch
        # size, you have two options leave the graph placeholders to None in the first dim, or set drop_reminder = True
        # In tf 2.0 was removed (not so sure, correct me if I am wrong)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        # We initialize our basic iterator
        iterator = dataset.make_one_shot_iterator()
        # get the next features every time it gets called
        features, labels = iterator.get_next()
        # batch dimension will be (64, 48, 48, 1)  labels (68, 7 or 8 depending on the data classes)
        return features, labels


if __name__ == '__main__':
    pass
