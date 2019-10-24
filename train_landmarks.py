import os
import glob

import dlib

import numpy as np
import tensorflow as tf

from lib.data_loader import _CNN_Data_Loader
from lib.config import ConfigReader, TrainNetConfig, DataConfig
from lib.CNNS.land_marks import _facenet

#NUM_PARALLEL_EXEC_UNITS = 4


# p = '/Users/miguelangelalbaacosta/Downloads/shape_predictor_68_face_landmarks.dat'
p = '/home/miguel_alba/data/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]


def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def create_batch_landmarks(crops):
    landmarks = []
    for ex in crops:
        lands = get_landmarks(ex[:, :, 0], face_rects)
        landmarks.append(lands)
    batch = np.array(landmarks).astype(np.float32)
    return batch


def train(conf_path):
    conf_path = conf_path
    config_reader = ConfigReader(conf_path)
    train_config = TrainNetConfig(config_reader.get_train_config())
    data_config = DataConfig(config_reader.get_train_config())

    out_dir = os.path.join(train_config.checkpoint_dir, 'models', train_config.name)
    train_log_dir = '{}/logs/train/'.format(out_dir)
    test_log_dir = '{}/logs/test/'.format(out_dir)

    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)

    net = _facenet(train_config)

    with tf.name_scope('input'):
        train_loader = _CNN_Data_Loader(data_config, name='train', training_mode=True, shuffle=True)
        train_image_batch, train_label_batch = train_loader._generate_batch()
        test_loader = _CNN_Data_Loader(data_config, name='test', training_mode=False, shuffle=False)  # default false
        test_image_batch, test_label_batch = test_loader._generate_batch()

    loss, accuracy = net.batch_model()
    train_op = net.optimize(loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

    try:
        for step in np.arange(train_config.max_step):
            train_image, train_label = sess.run([train_image_batch, train_label_batch])
            train_land_marks = create_batch_landmarks(train_image)
            assert train_label.shape[1] == data_config.n_classes
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                feed_dict={net.x: train_image, net.y: train_label,
                                                           net.landmark: train_land_marks})
            if step % 50 == 0 or step + 1 == train_config.max_step:
                print('===TRAIN===: Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: train_image, net.y: train_label,
                                                              net.landmark: train_land_marks})
                train_summary_writer.add_summary(summary_str, step)
            if step % 150 == 0 or step + 1 == train_config.max_step:
                val_image, val_label = sess.run([test_image_batch, test_label_batch])
                val_land_marks = create_batch_landmarks(val_image)
                plot_images = tf.summary.image('val_images_{}'.format(step % 200), val_image, 10)
                val_loss, val_acc, plot_summary = sess.run([loss, accuracy, plot_images],
                                                           feed_dict={net.x: val_image, net.y: val_label,
                                                                      net.landmark: val_land_marks})
                print('====VAL====: Step %d, val loss = %.4f, val accuracy = %.4f%%' % (step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: val_image, net.y: val_label,
                                                              net.landmark: val_land_marks})
                val_summary_writer.add_summary(summary_str, step)
                val_summary_writer.add_summary(plot_summary, step)
            if step % 2000 == 0 or step + 1 == train_config.max_step:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('===INFO====: Training completed, reaching the maximum number of steps')
    sess.close()
    return None


def _train_201909_():
    conf_path = '/Volumes/SSD_ML/facialexpr/lib/experiments/experiment_2.yml'
    train(conf_path)
    return None


def train_cloud():
    conf_path = '/home/miguel_alba/facialexpr/lib/experiments/experiment_cloud.yml'
    train(conf_path)
    return None


def main():
    #_train_201909_()
    train_cloud()


if __name__ == '__main__':
    main()
