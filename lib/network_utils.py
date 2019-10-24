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
        self.scope = {}

    def load(self, sess, saver, ckpt_path):
        print('Model checkpoint path: {}'.format(ckpt_path))
        try:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Done')
        except FileNotFoundError:
            raise 'Check your pretrained {:s}'.format(ckpt_path)
