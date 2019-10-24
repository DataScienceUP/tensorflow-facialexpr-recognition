import tensorflow as tf

from lib.network_utils import Net
from lib.CNNS import operations as ops

# this network was implemented without landmarks


class _facenet(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        # for consistent batches you set the None to the batch_size, None is to pass all data you want
        self.x = tf.placeholder(tf.float32, name='x', shape=[None,
                                                             self.config.image_height,
                                                             self.config.image_width,
                                                             self.config.image_depth])
        self.y = tf.placeholder(tf.int16, name='y', shape=[None,
                                                           self.config.n_classes])

    def get_summary(self):
        # we are using summaries to display tensorboard instead of writting a function to see the data and the
        # loss, accuracy graphs as well as the gradients you can check it on tf website
        return self.summary

    def cal_loss(self, logits, labels):
        # in order to display in tensorflow and know the operations you are doing you use a name and variable scope
        with tf.name_scope('loss') as scope:
            # common cross entropy, in this case the built in function computes the softmax for you so dont add it
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
        # You can choose and add several optimizers, keep in mind that if you want to visualize the gradients in order
        # to see if you are having vanishing gradient etc. you have to use a function called apply_gradients and not
        # minimize, check on internet
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

    # here you have several convolutional neural networks, feel free to experiment and pick the right one , remember
    # that the operations should be defined in the operations script, returning the
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
        dropout_f1 = ops.dropout(fc1, 0.50, is_training=self.is_training)

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

        fc1 = ops.fc('fc1', dropout4, out_nodes=1024)
        dropout_f1 = ops.dropout(fc1, 0.50, is_training=self.is_training)

        fc2 = ops.fc('fc2', dropout_f1, out_nodes=1024)
        dropout_f2 = ops.dropout(fc2, 0.50, is_training=self.is_training)

        self.logits = ops.fc('output_layer', dropout_f2, out_nodes=self.config.n_classes, act_type=None)

        loss = self.cal_loss(self.logits, self.y)
        accuracy = self.cal_accuracy(self.logits, self.y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        return loss, accuracy

    def res_net(self):

        conv1 = ops.conv('conv1', self.x, 16, kernel_size=[7, 7], stride=[1, 1, 1, 1], is_training=self.is_training)
        conv2 = ops.conv('conv2', conv1, 32, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_training=self.is_training)
        res1 = ops.residual_layer('residual_1', conv2, 32)
        pool1 = ops.pool('pool1', res1)
        res2 = ops.residual_layer('residual_2', pool1, 32)
        pool2 = ops.pool('pool2', res2)
        res3 = ops.residual_layer('residual_3', pool2, 32)
        conv3 = ops.conv('conv3', res3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_training=self.is_training)
        pool3 = ops.pool('pool3', conv3)
        lrn = ops.lrn('lrn1', pool3)

        dropout1 = ops.dropout(lrn, 0.50, is_training=self.is_training)

        fc1 = ops.fc('fc1', dropout1, out_nodes=128)
        dropout2 = ops.dropout(fc1, 0.40, is_training=self.is_training)
        batch_norm1 = ops.batch_normalization('batch_norm1', dropout2, is_training=self.is_training)

        fc2 = ops.fc('fc2', batch_norm1, out_nodes=512)
        batch_norm2 = ops.batch_normalization('batch_norm4', fc2, is_training=self.is_training)

        self.logits = ops.fc('output_layer', batch_norm2, out_nodes=self.config.n_classes, act_type=None)

        loss = self.cal_loss(self.logits, self.y)
        accuracy = self.cal_accuracy(self.logits, self.y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # train_op = self.optimize()
        return loss, accuracy

    def one_con_model(self):
        conv1_1 = ops.conv('conv1_1', self.x, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_training=self.is_training)
        conv1_2 = ops.conv('conv1_2', conv1_1, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_training=self.is_training)
        pool1 = ops.pool('pool1', conv1_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        lnr1 = ops.lrn('lrn1', pool1)

        conv2_1 = ops.conv('conv2_1', pool1, 32, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_training=self.is_training)
        conv2_2 = ops.conv('conv2_2', conv2_1, 32, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_training=self.is_training)
        lnr2 = ops.lrn('lrn2', conv2_2)

        concat1 = ops.layer_concat('concat1', [lnr1, lnr2])
        pool2 = ops.pool('pool2', concat1, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        conv3_1 = ops.conv('conv3_1', pool2, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_training=self.is_training)
        conv3_2 = ops.conv('conv3_2', conv3_1, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_training=self.is_training)
        pool3 = ops.pool('pool3', conv3_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        lnr3 = ops.lrn('lrn3', pool3)

        dropout1 = ops.dropout(lnr3, 0.60, is_training=self.is_training)

        fc1 = ops.fc('fc1', dropout1, out_nodes=128)
        dropout2 = ops.dropout(fc1, 0.40, is_training=self.is_training)
        batch_norm1 = ops.batch_normalization('batch_norm1', dropout2, is_training=self.is_training)

        fc2 = ops.fc('fc2', batch_norm1, out_nodes=512)
        batch_norm2 = ops.batch_normalization('batch_norm4', fc2, is_training=self.is_training)

        self.logits = ops.fc('output_layer', batch_norm2, out_nodes=self.config.n_classes, act_type=None)

        loss = self.cal_loss(self.logits, self.y)
        accuracy = self.cal_accuracy(self.logits, self.y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # train_op = self.optimize()
        return loss, accuracy

    def fex_net(self):
        conv1 = ops.conv('conv1', self.x, 64, kernel_size=[5, 5], stride=[1, 2, 2, 1], padding='SAME',
                         is_training=self.is_training)
        pool1 = ops.pool('pool1', conv1, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        norm1 = ops.lrn('local_response', pool1)

        conv2a = ops.conv('conv2a', norm1, 96, kernel_size=[1, 1], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        conv2b = ops.conv('conv2b', conv2a, 208, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        pool2a = ops.pool('pool2a', norm1, kernel=[1, 3, 3, 1], stride=[1, 1, 1, 1], is_max_pool=True)
        conv2c = ops.conv('conv2c', pool2a, 64, kernel_size=[1, 1], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        concat1 = ops.layer_concat('concat1', inputs=[conv2b, conv2c])
        pool2b = ops.pool('pool2b', concat1, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        conv3a = ops.conv('conv3a', pool2b, 96, kernel_size=[1, 1], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        conv3b = ops.conv('conv3b', conv3a, 208, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        pool3a = ops.pool('pool3a', pool2b, kernel=[1, 3, 3, 1], stride=[1, 1, 1, 1], is_max_pool=True)
        conv3c = ops.conv('conv3c', pool3a, 64, kernel_size=[1, 1], stride=[1, 1, 1, 1], padding='SAME',
                          is_training=self.is_training)
        concat2 = ops.layer_concat('concat2', inputs=[conv3b, conv3c])
        # pool3b = ops.pool('pool3b', concat2, kernel=[1, 3, 3, 1], stride=[1, 1, 1, 1], is_max_pool=True)

        print(conv3a.get_shape())
        print(conv3b.get_shape())
        print(pool3a.get_shape())
        print(conv3c.get_shape())
        print(concat2.get_shape())
        # print(pool3b.get_shape())

        self.logits = ops.fc('output_layer', concat2, out_nodes=self.config.n_classes, act_type='softmax')

        loss = self.cal_loss(self.logits, self.y)
        accuracy = self.cal_accuracy(self.logits, self.y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # train_op = self.optimize()
        return loss, accuracy

    def build_model(self):
        self.conv1_1 = ops.conv('conv1_1', self.x, 32, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        self.conv1_2 = ops.conv('conv1_2', self.conv1_1, 32, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        self.pool1 = ops.pool('pool1', self.conv1_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        self.pool1_norm1 = ops.lrn('pool1_norm1', self.pool1)

        self.conv2_1 = ops.conv('conv2_1', self.pool1_norm1,  64, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        self.conv2_2 = ops.conv('conv2_2', self.conv2_1, 64, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        self.batch_norm1 = ops.batch_normalization('batch_norm1', self.conv2_2, is_training=self.is_training)
        self.pool2 = ops.pool('pool2', self.batch_norm1, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv3_1 = ops.conv('conv3_1', self.pool2, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        self.conv3_2 = ops.conv('conv3_2', self.conv3_1, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        self.batch_norm2 = ops.batch_normalization('batch_norm2', self.conv3_2, is_training=self.is_training)
        self.pool3 = ops.pool('pool3', self.batch_norm2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.fc1 = ops.fc('fc1', self.pool3, out_nodes=1024)
        self.batch_norm3 = ops.batch_normalization('batch_norm3', self.fc1, is_training=self.is_training)
        self.dropout1 = ops.dropout(self.batch_norm3, 0.50, is_training=self.is_training)

        self.fc2 = ops.fc('fc2', self.dropout1, out_nodes=512)
        self.batch_norm4 = ops.batch_normalization('batch_norm4', self.fc2, is_training=self.is_training)

        self.fc3 = ops.fc('fc3', self.batch_norm4, out_nodes=256)
        self.dropout3 = ops.dropout(self.fc3, 0.50, is_training=self.is_training)

        self.logits = ops.fc('fc_output', self.dropout3, out_nodes=self.config.n_classes, act_type=None)

        loss = self.cal_loss(self.logits, self.y)
        accuracy = self.cal_accuracy(self.logits, self.y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # train_op = self.optimize()
        return loss, accuracy