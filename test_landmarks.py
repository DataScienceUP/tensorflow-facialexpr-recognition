import os
import glob

import dlib

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt

import tensorflow as tf
from tqdm import tqdm

import util

from prep.facialexpr import encode_relabels
from lib.config import ConfigReader, TestNetConfig, TrainNetConfig, DataConfig
from lib.CNNS.land_marks import _facenet
from train_landmarks import get_landmarks, create_batch_landmarks


#p = '/Users/miguelangelalbaacosta/Downloads/shape_predictor_68_face_landmarks.dat'
p = '/home/miguel_alba/data/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]


def prep_crops(data):
    prep_data = []
    for ex in data:
        crop = ex['crop'].copy()
        crop = crop[..., None]
        ex['crop'] = crop
        prep_data.append(ex)
    return prep_data


def relab_one_hot(labels):
    label_dict = {
        0: 'anger',
        1: 'contempt',
        2: 'disgust',
        3: 'fear',
        4: 'happiness',
        5: 'neutral',
        6: 'sadness',
        7: 'surprise'}
    num = []
    for i, img in enumerate(labels):
        num.append(label_dict[np.argmax(labels[i], axis=-1)])
    return np.asarray(num)


def confusion_matrix(groundtruth, new_pred, out_dir, label_names=None):
    plt.style.use('ggplot')
    confusion = sklearn.metrics.confusion_matrix(groundtruth, new_pred)
    if label_names is not None:
        labels = label_names
    else:
        labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax = sns.heatmap(confusion, ax=ax, cmap=plt.cm.Blues, annot=True)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.title('Confusion matrix (Validation set)')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    #
    model_accuracy = np.trace(confusion) / sum(sum(confusion))
    print("model accuracy: ", model_accuracy)
    #
    precision = np.diagonal(confusion) / np.sum(confusion, axis=0)
    print(pd.DataFrame({'label': labels, 'Precision': precision}))
    #
    recall = np.diagonal(confusion) / np.sum(confusion, axis=1)
    print(pd.DataFrame({'label': labels, 'Recall': recall}))

    results = dict(model_accuracy=model_accuracy,
                   classes=labels,
                   presicion=precision.tolist(),
                   recall=recall.tolist())
    util.spit_json('{}/result.json'.format(out_dir), results)
    return None


def evaluation(conf_path):
    conf_path = conf_path
    config_reader = ConfigReader(conf_path)
    train_config = TrainNetConfig(config_reader.get_train_config())
    test_config = TestNetConfig(config_reader.get_test_config())
    data_config = DataConfig(config_reader.get_train_config())

    ckpt_path = '{}/logs/train'.format(os.path.join(test_config.model_path, 'models', train_config.name))
    eval_files = glob.glob('{}/eval_*.pz'.format(data_config.eval_dir))

    eval_data = []
    for fn in tqdm(eval_files):
        _data = util.load1(fn)
        _data = prep_crops(_data)
        eval_data.extend(_data)

    net = _facenet(test_config)
    net.batch_model()

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    net.load(sess, saver, ckpt_path)

    results = []
    try:
        for ex in tqdm(eval_data):
            crop = [ex['crop'].copy()]
            landmark = np.array([get_landmarks(ex['crop'][:, :, 0], face_rects)]).astype(np.float32)
            crop = np.array(crop).astype(np.float32)
            ground_truth = encode_relabels(ex['label'])
            predicted = sess.run(tf.nn.softmax(net.logits), feed_dict={net.x: crop, net.landmark: landmark})
            results.append([predicted, ground_truth])
    except tf.errors.OutOfRangeError:
        print('===INFO====: Test completed, all crops were successfully evaluated')
    sess.close()
    results_dir = os.path.join(test_config.model_path, 'results', train_config.name)
    util.mdir(results_dir)
    util.save1('{}/predictions.pz'.format(results_dir), results)
    predictions = [relab_one_hot(p[0]) for p in results]
    labels = [relab_one_hot([p[1]]) for p in results]
    confusion_matrix(labels, predictions, results_dir)
    return results


def _test_201909_():
    conf_path = '/Volumes/SSD_ML/facialexpr/lib/experiments/experiment_2.yml'
    results = evaluation(conf_path)
    return None


def cloud_tesing():
    conf_path = '/home/miguel_alba/facialexpr/lib/experiments/experiment_cloud.yml'
    results = evaluation(conf_path)
    return None


def main():
    #_test_201909_()
    cloud_tesing()


if __name__ == '__main__':
    main()
