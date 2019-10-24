import os
import sys
import glob
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

import utils


def get_labels(expr):
    """
    This oncodings are not necessary at all as we are going to use input pipeline performance with the tf.dataset api
    in the API you can use internal tensorflow functions to encode the data
    this function was created to be called in evaluation as we are not going to feed the data inside of the model using
    tf.dataset api
    :param expr:
    :return:
    """
    if expr is 0:
        return [1, 0, 0, 0, 0, 0, 0]
    if expr is 1:
        return [0, 1, 0, 0, 0, 0, 0]
    if expr is 2:
        return [0, 0, 1, 0, 0, 0, 0]
    if expr is 3:
        return [0, 0, 0, 1, 0, 0, 0]
    if expr is 4:
        return [0, 0, 0, 0, 1, 0, 0]
    if expr is 5:
        return [0, 0, 0, 0, 0, 1, 0]
    if expr is 6:
        return [0, 0, 0, 0, 0, 0, 1]


def encode_relabels(expr):
    """
    one hot encoding for label exressions in the fer+ relabeling
    :param expr:
    :return:
    """
    if expr is 0:
        return [1, 0, 0, 0, 0, 0, 0, 0]
    if expr is 1:
        return [0, 1, 0, 0, 0, 0, 0, 0]
    if expr is 2:
        return [0, 0, 1, 0, 0, 0, 0, 0]
    if expr is 3:
        return [0, 0, 0, 1, 0, 0, 0, 0]
    if expr is 4:
        return [0, 0, 0, 0, 1, 0, 0, 0]
    if expr is 5:
        return [0, 0, 0, 0, 0, 1, 0, 0]
    if expr is 6:
        return [0, 0, 0, 0, 0, 0, 1, 0]
    if expr is 7:
        return [0, 0, 0, 0, 0, 0, 0, 1]


def preprocess_fer_dataset(data_path):
    """
    :param data_path: data path that contains the original 2013 fex csv
    :return: it returns a tuple containing list of dictionaries for train, test, eval
    """
    if str(data_path).endswith('/'):
        data_path = data_path[:-1]

    facialexpr_filename = os.path.join(data_path, 'fer2013.csv')
    print('Recovering raw dataframe')
    raw_data = pd.read_csv(facialexpr_filename)
    print('Number of experiment observations: {}'.format(raw_data.shape[0]))
    #
    new_data = []
    for idx, row in tqdm(raw_data.iterrows()):
        encoded_label = get_labels(row['emotion'])
        recover_crop = np.fromstring(str(row['pixels']), dtype=np.uint8, sep=' ').reshape((48, 48))
        example = {
            'crop': recover_crop,
            'label': row['emotion'],
            'encoded_label': encoded_label,
            'data_usage': row['Usage']
        }
        new_data.append(example)
    #
    train = [ex for ex in new_data if ex['data_usage'] == 'Training']
    test = [ex for ex in new_data if ex['data_usage'] == 'PrivateTest']
    evaluation = [ex for ex in new_data if ex['data_usage'] == 'PublicTest']
    print('Number of observations for training {}'.format(len(train)))
    print('Number of observations for testing {}'.format(len(test)))
    print('Number of observations for evaluation {}'.format(len(evaluation)))
    return train, test, evaluation


def generate_chunks(data, name, out_dir, chunk_lenght=1000):
    """
    This function is made with the purpose of transforming list of dictionaries into chunks of data for movility purposes
    in big scale training strategies, you train with several files as you can not load everything in memory at once
    (ex. keras, pandas, etc)
    :param data: list of dictionaries
    :param name: name for the data folder
    :param out_dir: output directory
    :param chunk_lenght: lenght of the zip files
    :return:None (just to keep syntax)
    """
    parts = [(k * chunk_lenght) for k in range(len(data)) if (k * chunk_lenght) < len(data)]
    print('files generated for {} set, with {} chunks'.format(name, len(parts)))
    if not os.path.exists(out_dir):
        utils.mdir(out_dir)
    for i, j in enumerate(parts):
        new_data = data[j:(j + chunk_lenght)]
        random.seed(1)
        new_data = random.sample(new_data, len(new_data))
        fn = '{}_{}-{}.data'.format(name, format(i + 1, '03d'), len(parts))
        utils.save(os.path.join(out_dir, fn), new_data)
    print('training/evaluation batches saved in: {}'.format(out_dir))
    return None


def generate_example_sets(data_path, name, chunk_lenght=1000):
    """
    it saves all data in a selected folder with an specific name for train, test, eval
    :param data_path:
    :param name:
    :param chunk_lenght:
    :return:
    """
    train, test, evaluation = preprocess_fer_dataset(data_path)
    out_dir = os.path.join(data_path, name)
    utils.mdir(out_dir)
    generate_chunks(train, name='train', out_dir=out_dir, chunk_lenght=chunk_lenght)
    generate_chunks(test, name='test', out_dir=out_dir, chunk_lenght=chunk_lenght)
    generate_chunks(evaluation, name='eval', out_dir=out_dir, chunk_lenght=chunk_lenght)
    return None


def debug_preprocessing(data_path, name):
    """
    take one of the files saved and test if the information is okey, you can do it on a jupyter or using an iteractive
    IDE
    :param data_path:
    :param name:
    :return:
    """
    data_fns = glob.glob('{}/{}_*.data'.format(data_path, name))
    face_data = []
    for fn in tqdm(data_fns):
        data = utils.load(fn)
        face_data.extend(data)
    print(face_data[0])
    return None

# tf records part


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _process_examples(example_data, filename: str, channels=1):
    """
    :param example_data: takes the list of dictionaries and transform them into Tf records, this is an special format
    of tensorflow data that makes your life easier in tf 1.x and 2.0 saving the data and load it in our training loop
    (WARNING: You have to take care of the encoding of features to not have problems when loading the data, this means
    taking into consideration that images are int or float)
    :param filename: output filename
    :param channels: number of channels of the image (RGB=3), grayscale=!
    :return: None
    """
    print(f'Processing {filename} data')
    dataset_length = len(example_data)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index, ex in enumerate(example_data):
            sys.stdout.write(f"\rProcessing sample {index + 1} of {dataset_length}")
            sys.stdout.flush()
            crop = ex['crop'].flatten()
            crop = crop.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(ex['crop'].shape[0]),
                'width': _int64_feature(ex['crop'].shape[1]),
                'depth': _int64_feature(channels),
                'label': _int64_feature(ex['label']), # _encode_labels
                'crop': _bytes_feature(crop)
            }))
            writer.write(example.SerializeToString())
        print()
    return None


def transform_to_tfrecords(data_path, name: str, channels=None, num_records=10):
    """
    This function is to iterate over all compressed files that we have and transform them in tf record data
    :param data_path: preprocessing folder
    :param name: Only train or test
    :param channels: 1
    :param num_records: number of files
    :return:
    """
    if str(data_path).endswith('/'):
        data_path = data_path[:-1]
        #
    if channels is None:
        channels = 1
    filenames = glob.glob('{}/{}_*.pz'.format(data_path, name))
    facialexpr_data = []
    for fn in tqdm(filenames):
        data = utils.load(fn)
        facialexpr_data.extend(data)
    #
    print('number of observations for {}:{}'.format(name, len(facialexpr_data)))
    samples_per_tf_record = len(facialexpr_data) // num_records
    tf_parts = [(k * samples_per_tf_record) for k in range(len(facialexpr_data)) if
                (k * samples_per_tf_record) < len(facialexpr_data)]
    utils.mdir(os.path.join(data_path, 'training'))
    for i, j in enumerate(tf_parts):
        out_fn = os.path.join(data_path, 'training', '{}_{:03d}-{:03d}.tfrecord'.format(name, i+1, num_records))
        _process_examples(facialexpr_data[j:(j + samples_per_tf_record)], out_fn, channels=channels)
    return None


def _20190910_preprocessing():
    data_path = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013'
    name = '20190910_facialexpr_fer2013_format'
    generate_example_sets(data_path, name)


def _20190911_generate_tf_records():
    data_path = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013/20190910_facialexpr_fer2013_format'
    transform_to_tfrecords(data_path, name='train', num_records=10)
    transform_to_tfrecords(data_path, name='test', num_records=2)
    return None


if __name__ == '__main__':
    _20190910_preprocessing()
    _20190911_generate_tf_records()
    pass