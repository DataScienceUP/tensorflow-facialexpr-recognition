import os
import glob
import copy

import pandas as pd
import numpy as np

import tensorflow as tf
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils

from prep.facialexpr import encode_relabels
from lib.config import ConfigReader, TestNetConfig, TrainNetConfig, DataConfig
from lib.CNNS.architecture import _facenet


def prep_crops(data):
    """
    :param data: batch of list of dicts
    :return: observations with an image of (h,w,1)
    """
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
    utils.spit_json('{}/result.json'.format(out_dir), results)
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
        _data = utils.load(fn)
        _data = prep_crops(_data)
        eval_data.extend(_data)

    # select the model vgg13, batch... etc
    net = _facenet(test_config)
    net.batch_model()

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # load the model
    net.load(sess, saver, ckpt_path)

    # with the loaded model and predict using one observation at the time (1,h,w,1), you can change it for the batch
    # that you want re arranging the loop
    results = []
    try:
        for ex in tqdm(eval_data):
            crop = [ex['crop'].copy()]
            crop = np.array(crop).astype(np.float32)
            ground_truth = encode_relabels(ex['label'])
            predicted = sess.run(tf.nn.softmax(net.logits), feed_dict={net.x: crop})
            # save pred and ground truth
            results.append([predicted, ground_truth])
    except tf.errors.OutOfRangeError:
        print('===INFO====: Test completed, all crops were successfully evaluated')
    sess.close()
    results_dir = os.path.join(test_config.model_path, 'results', train_config.name)
    utils.mdir(results_dir)
    utils.save('{}/predictions.data'.format(results_dir), results)
    # computing configusion metrics
    predictions = [relab_one_hot(p[0]) for p in results]
    labels = [relab_one_hot([p[1]]) for p in results]
    confusion_matrix(labels, predictions, results_dir)
    return results

# This part is under construction  Local Interpretability Model Agnostic Explanations (LIME)
# to see how is the prediction


def perturb_image(img, perturbation, segments):
      active_pixels = np.where(perturbation == 1)[0]
      mask = np.zeros(segments.shape)
      for active in active_pixels:
          mask[segments == active] = 1
      perturbed_image = copy.deepcopy(img)
      perturbed_image = perturbed_image*mask[:,:,np.newaxis]
      return perturbed_image


def LIME(conf_path):
    from sklearn.linear_model import LinearRegression
    import skimage.segmentation
    conf_path = conf_path
    config_reader = ConfigReader(conf_path)
    train_config = TrainNetConfig(config_reader.get_train_config())
    test_config = TestNetConfig(config_reader.get_test_config())
    data_config = DataConfig(config_reader.get_train_config())

    ckpt_path = '{}/logs/train'.format(os.path.join(test_config.model_path, 'models', train_config.name))
    eval_files = glob.glob('{}/eval_*.pz'.format(data_config.eval_dir))

    eval_data = []
    for fn in tqdm(eval_files):
        _data = utils.load(fn)
        _data = prep_crops(_data)
        eval_data.extend(_data)

    net = _facenet(test_config)
    net.vgg13()

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    net.load(sess, saver, ckpt_path)

    crop = eval_data[10]['crop']
    label = eval_data[10]['label']
    img = np.concatenate((crop, crop, crop), axis=2)
    superpixels = skimage.segmentation.quickshift(img, kernel_size=1, max_dist=10, ratio=0.1)
    num_superpixels = np.unique(superpixels).shape[0]
    num_perturb = 150
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    ground_truth = encode_relabels(label)
    predictions = []
    try:
        for per in tqdm(perturbations):
            per_image = perturb_image(img, per, superpixels)
            per_image = [per_image]
            per_image = np.array(per_image).astype(np.float32)
            predicted = sess.run(tf.nn.softmax(net.logits), feed_dict={net.x: per_image})
            predictions.append([predicted])
    except tf.errors.OutOfRangeError:
        print('===INFO====: Test completed, all crops were successfully evaluated')
    predictions = np.array(predictions)

    original_image = np.ones(num_superpixels)[np.newaxis, :]  # Perturbation with all superpixels enabled
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()

    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function

    # I am working on adding some Local Interpretability Model Agnostic Explanations
    raise NotImplementedError
    

def _test_201909_():
    conf_path = '/Volumes/SSD_ML/facialexpr/lib/experiments/experiment_2.yml'
    results = evaluation(conf_path)
    return None


def main():
    _test_201909_()


if __name__ == '__main__':
    main()
