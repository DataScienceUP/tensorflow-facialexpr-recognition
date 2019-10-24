import tensorflow as tf

from lib.network_utils import Net
from lib.CNNS import operations as ops


class _facenet(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        # consistent batches
        self.x = tf.placeholder(tf.float32, name='x', shape=[None,
                                                             self.config.image_height,
                                                             self.config.image_width,
                                                             self.config.image_depth])
        self.y = tf.placeholder(tf.int32, name='y', shape=[None,
                                                           self.config.n_classes])

        self.landmark = tf.placeholder(tf.float32, name='landmark', shape=[None, 68, 2])

    def get_summary(self):
        return self.summary

    def cal_loss(self, logits, labels):
        with tf.name_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross-entropy')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            return loss

    def cal_accuracy(self, logits, labels):
        with tf.name_scope('accuracy') as scope:
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.cast(correct, tf.float32)
            accuracy = tf.reduce_mean(correct) * 100.0
            return accuracy

    def optimize(self, loss):
        with tf.name_scope('optimizer'):
            if self.config.type_optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.type_optimizer == 'AdamW':
                optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=self.config.learning_rate,
                                                          weight_decay=self.config.weight_decay)
            elif self.config.type_optimizer == 'Momentum':
                # momentum = -self.config.batch_size/np.log(0.9)
                momentum = 0.9
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=momentum)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            train_op = optimizer.minimize(loss, global_step=self.global_step_tensor)
            return train_op

    def batch_model(self):
        re_input = tf.image.resize_nearest_neighbor(self.x, size=(64, 64))
        conv1_1 = ops.conv('conv1_1', re_input, 64, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        conv1_2 = ops.conv('conv1_2', conv1_1, 64, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        bn1_1 = ops.batch_normalization('bn1_1', conv1_2, self.is_training)
        pool1_1 = ops.pool('pool1_1', bn1_1)
        dropout1 = ops.dropout(pool1_1, 0.5, is_training=self.is_training)

        conv2_1 = ops.conv('conv2_1', dropout1, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn2_1 = ops.batch_normalization('bn2_1', conv2_1, self.is_training)
        conv2_2 = ops.conv('conv2_2', bn2_1, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn2_2 = ops.batch_normalization('bn2_2', conv2_2, self.is_training)
        conv2_3 = ops.conv('conv2_3', bn2_2, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn2_3 = ops.batch_normalization('bn2_3', conv2_3, self.is_training)
        pool2_1 = ops.pool('pool2_1', bn2_3)
        dropout2 = ops.dropout(pool2_1, 0.5, is_training=self.is_training)

        conv3_1 = ops.conv('conv3_1', dropout2, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn3_1 = ops.batch_normalization('bn3_1', conv3_1, self.is_training)
        conv3_2 = ops.conv('conv3_2', bn3_1, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn3_2 = ops.batch_normalization('bn3_2', conv3_2, self.is_training)
        conv3_3 = ops.conv('conv3_3', bn3_2, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn3_3 = ops.batch_normalization('bn3_3', conv3_3, self.is_training)
        pool3_1 = ops.pool('pool3_1', bn3_3)
        dropout3 = ops.dropout(pool3_1, 0.5, is_training=self.is_training)

        conv4_1 = ops.conv('conv4_1', dropout3, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn4_1 = ops.batch_normalization('bn4_1', conv4_1, self.is_training)
        conv4_2 = ops.conv('conv4_2', bn4_1, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn4_2 = ops.batch_normalization('bn4_2', conv4_2, self.is_training)
        conv4_3 = ops.conv('conv4_3', bn4_2, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                           is_training=self.is_training)
        bn4_3 = ops.batch_normalization('bn4_3', conv4_3, self.is_training)
        pool4_1 = ops.pool('pool4_1', bn4_3)
        dropout4 = ops.dropout(pool4_1, 0.5, is_training=self.is_training)

        fc1 = ops.fc('fc1', dropout4, out_nodes=512)
        fcl1 = ops.land_layer('land_mark1', self.landmark, out_nodes=512)
        # we concatenate the landmarks with this layer so it get expanded to 1024
        concat = ops.concat_landmarks('landmark_adding', [fc1, fcl1])

        dropout_f1 = ops.dropout(concat, 0.50, is_training=self.is_training)

        fc2 = ops.fc('fc2', dropout_f1, out_nodes=256)
        dropout_f2 = ops.dropout(fc2, 0.50, is_training=self.is_training)

        fc3 = ops.fc('fc3', dropout_f2, out_nodes=64)
        dropout_f3 = ops.dropout(fc3, 0.50, is_training=self.is_training)

        self.logits = ops.fc('output_layer', dropout_f3, out_nodes=self.config.n_classes, act_type=None)

        loss = self.cal_loss(self.logits, self.y)
        accuracy = self.cal_accuracy(self.logits, self.y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        return loss, accuracy

    # try this architecture
    def vgg13(self):
        re_input = tf.image.resize_nearest_neighbor(self.x, size=(64, 64))
        conv1 = ops.conv('conv1', re_input, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        conv2 = ops.conv('conv2', conv1, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        pool1 = ops.pool('pool1', conv2)
        dropout1 = ops.dropout(pool1, 0.25, is_training=self.is_training)

        conv3 = ops.conv('conv3', dropout1, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        conv4 = ops.conv('conv4', conv3, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        pool2 = ops.pool('pool2', conv4)
        dropout2 = ops.dropout(pool2, 0.25, is_training=self.is_training)

        conv6 = ops.conv('conv6', dropout2, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        conv7 = ops.conv('conv7', conv6, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        conv8 = ops.conv('conv8', conv7, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        pool3 = ops.pool('pool3', conv8)
        dropout3 = ops.dropout(pool3, 0.25, is_training=self.is_training)

        conv9 = ops.conv('conv9', dropout3, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        conv10 = ops.conv('conv10', conv9, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        conv11 = ops.conv('conv11', conv10, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',  is_training=self.is_training)
        pool4 = ops.pool('pool4', conv11)
        dropout4 = ops.dropout(pool4, 0.25, is_training=self.is_training)

        fc1 = ops.fc('fc1', dropout4, out_nodes=512)
        fcl1 = ops.land_layer('land_mark1', self.landmark, out_nodes=512)
        # we concatenate the landmarks with this layer so it get expanded to 1024
        concat = ops.concat_landmarks('landmark_adding', [fc1, fcl1])
        dropout_f1 = ops.dropout(concat, 0.50, is_training=self.is_training)

        fc2 = ops.fc('fc2', dropout_f1, out_nodes=1024)
        dropout_f2 = ops.dropout(fc2, 0.50, is_training=self.is_training)

        self.logits = ops.fc('output_layer', dropout_f2, out_nodes=self.config.n_classes, act_type=None)

        loss = self.cal_loss(self.logits, self.y)
        accuracy = self.cal_accuracy(self.logits, self.y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        return loss, accuracy
