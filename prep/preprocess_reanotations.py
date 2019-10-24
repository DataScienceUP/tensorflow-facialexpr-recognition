import os
import glob

import scipy.misc
import numpy as np

import pandas as pd
from tqdm import tqdm

import utils

from prep.facialexpr import generate_chunks
from prep.facialexpr import transform_to_tfrecords


def get_relabel(label):
    if label == 'anger':
        return 0
    if label == 'contempt':
        return 1
    if label == 'disgust':
        return 2
    if label == 'fear':
        return 3
    if label == 'happiness':
        return 4
    if label == 'neutral':
        return 5
    if label == 'sadness':
        return 6
    if label == 'surprise':
        return 7


def preprocess_relabeled_data(data_path):
    """
    This function takes the preprocessing made by microsoft, which script is called by reanotations, it uses parsed
    arguments, you have to follow some instructions presented in their github to create the new csv file with
    the relabeling
    :param data_path: new_relabeled csv
    :return:
    """
    FERTRAIN = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013/data/FER2013Train'
    FERVALID = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013/data/FER2013Valid'
    FERTEST = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013/data/FER2013Test'

    relabeling = pd.read_csv('{}/fer2013new.csv'.format(data_path))
    print('number of observations: {}'.format(relabeling.shape))
    relabeling = relabeling.dropna()
    meta_dict = relabeling.to_dict('index')
    asa_format = [v for k, v in meta_dict.items()]

    facial_expressions = list(relabeling.columns)[2:-2]

    data = []
    for i, ex in enumerate(tqdm(asa_format)):
        if ex['Usage'] == 'Training':
            img_filename = os.path.join(FERTRAIN, ex['Image name'])
        if ex['Usage'] == 'PublicTest':
            img_filename = os.path.join(FERVALID, ex['Image name'])
        if ex['Usage'] == 'PrivateTest':
            img_filename = os.path.join(FERTEST, ex['Image name'])

        fex_dict = {e: ex[e] for e in facial_expressions}
        relabel = max(fex_dict, key=fex_dict.get)

        ex['label'] = get_relabel(relabel)
        ex['crop'] = scipy.misc.imread(img_filename).astype(np.uint8)
        data.append(ex)

    train = [ex for ex in data if ex['Usage'] == 'Training']
    test = [ex for ex in data if ex['Usage'] == 'PublicTest']
    evaluation = [ex for ex in data if ex['Usage'] == 'PrivateTest']
    print('Number of observations for training {}'.format(len(train)))
    print('Number of observations for testing {}'.format(len(test)))
    print('Number of observations for evaluation {}'.format(len(evaluation)))
    return train, test, evaluation


def generate_relabel_sets(data_path, name, chunk_lenght=1000):
    train, test, evaluation = preprocess_relabeled_data(data_path)
    out_dir = os.path.join(data_path, name)
    utils.mdir(out_dir)
    generate_chunks(train, name='train', out_dir=out_dir, chunk_lenght=chunk_lenght)
    generate_chunks(test, name='test', out_dir=out_dir, chunk_lenght=chunk_lenght)
    generate_chunks(evaluation, name='eval', out_dir=out_dir, chunk_lenght=chunk_lenght)


def _20190924_tfrecords():
    data_path = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013/20190924_relabeled_fer_reanotations'
    transform_to_tfrecords(data_path, name='train', num_records=10)
    transform_to_tfrecords(data_path, name='test', num_records=2)
    return None


if __name__ == '__main__':
    data_path = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013'
    generate_relabel_sets(data_path, name='20190924_relabeled_fer_reanotations')
    _20190924_tfrecords()
    pass