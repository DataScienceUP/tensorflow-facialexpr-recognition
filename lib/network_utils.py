import tensorflow as tf

from lib.config import DataConfig, TrainNetConfig


class Net:
    def __init__(self, cfg_):
        """
        The most important part in this structure is the load function as it will help us to restore
        the model and predict on new observations
        :type cfg_: Config
        :param cfg_:
        """
        self.config = cfg_
        self.is_training = True if self.config.mode == 'TRAIN' else False
        self.saver = None
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.scope = {}

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def load(self, sess, saver, ckpt_path):
        print('Model checkpoint path: {}'.format(ckpt_path))
        try:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Done')
        except FileNotFoundError:
            raise 'Check your pretrained {:s}'.format(ckpt_path)

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')