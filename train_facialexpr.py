import os
import glob

import numpy as np
import tensorflow as tf

from lib.data_loader import _CNN_Data_Loader
from lib.config import ConfigReader, TrainNetConfig, DataConfig
from lib.CNNS.architecture import _facenet

# This code can change a bit depending on your computer or server specifications, for example adding GPU support
# os.environ["CUDA_VISIBLE_DEVICES"] = 1
NUM_PARALLEL_EXEC_UNITS = 4


def train(conf_path):
    # read configurations
    conf_path = conf_path
    config_reader = ConfigReader(conf_path)
    train_config = TrainNetConfig(config_reader.get_train_config())
    data_config = DataConfig(config_reader.get_train_config())

    # setting paths to save model
    out_dir = os.path.join(train_config.checkpoint_dir, 'models', train_config.name)
    train_log_dir = '{}/logs/train/'.format(out_dir)
    test_log_dir = '{}/logs/test/'.format(out_dir)

    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)

    # this calls the super class
    net = _facenet(train_config)

    # here we call the data loaders to generate batches for train and test
    with tf.name_scope('input'):
        train_loader = _CNN_Data_Loader(data_config, name='train', training_mode=True, shuffle=True)
        train_image_batch, train_label_batch = train_loader._generate_batch()
        test_loader = _CNN_Data_Loader(data_config, name='test', training_mode=False, shuffle=False)  # default false
        test_image_batch, test_label_batch = test_loader._generate_batch()

    # we call first the network and we call the training opt
    # this net.batch_model() may change  depending on the architecture you want to feed
    loss, accuracy = net.batch_model()
    train_op = net.optimize(loss)

    # this is for enhace CPU training sessions
    # config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
    #                        inter_op_parallelism_threads=2,
    #                        allow_soft_placement=True,
    #                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

    # os.environ["OMP_NUM_THREADS"] = "4"
    # os.environ["KMP_BLOCKTIME"] = "30"
    # os.environ["KMP_SETTINGS"] = "1"
    # os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

    # In case of GPU use:
    # call os.environ["CUDA_VISIBLE_DEVICES"] = 1 after the modules call,
    # >here< you use :
    # config = tf.ConfigProto()
    # sess = tf.Session(config=config)

    # Initialize the summaries to write in tensorboard
    summary_op = tf.summary.merge_all()
    # initialize the saver and the model variables
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    #sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

    # start training loop
    try:
        for step in np.arange(train_config.max_step):
            # here first we generate the batches of images and labels
            train_image, train_label = sess.run([train_image_batch, train_label_batch])
            assert train_label.shape[1] == data_config.n_classes
            # then feed them into the graph, running the training optimization and extracting values of loss and acc
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                feed_dict={net.x: train_image, net.y: train_label})
            if step % 50 == 0 or step + 1 == train_config.max_step:
                # every 50 iterations we print and save the summaries
                print('===TRAIN===: Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: train_image, net.y: train_label})
                train_summary_writer.add_summary(summary_str, step)
            if step % 150 == 0 or step + 1 == train_config.max_step:
                # every 150 iterations we generate a validation batch and save some 10 images
                val_image, val_label = sess.run([test_image_batch, test_label_batch])
                plot_images = tf.summary.image('val_images_{}'.format(step % 200), val_image, 10)
                val_loss, val_acc, plot_summary = sess.run([loss, accuracy, plot_images],
                                                           feed_dict={net.x: val_image, net.y: val_label})
                print('====VAL====: Step %d, val loss = %.4f, val accuracy = %.4f%%' % (step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: val_image, net.y: val_label})
                val_summary_writer.add_summary(summary_str, step)
                val_summary_writer.add_summary(plot_summary, step)
            if step % 2000 == 0 or step + 1 == train_config.max_step:
                # every 2000 steps we save a model, this to control problems that might let you stop training, then you
                # can reload the latest model and keep training
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('===INFO====: Training completed, reaching the maximum number of steps')
    sess.close()
    return None


def _train_201909_():
    # write all parameters in the configuration yml
    conf_path = '/Volumes/SSD_ML/facialexpr/lib/experiments/experiment_2.yml'
    train(conf_path)
    return None


def main():
    _train_201909_()


if __name__ == '__main__':
    main()
