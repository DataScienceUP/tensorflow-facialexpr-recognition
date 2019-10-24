import tensorflow as tf


def conv(name, input, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='VALID', is_training=True):
    in_channels = input.get_shape()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name='weights',
                            trainable=is_training,
                            shape=[kernel_size[0], kernel_size[1],
                                   in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',
                            trainable=is_training,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, w, stride, padding=padding, name='conv')
        conv = tf.nn.bias_add(conv, b, name='bias_add')
        out = tf.nn.relu(conv, name='relu')
        return out


def pool(name, input, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    with tf.name_scope(name):
        if is_max_pool:
            bottom = tf.nn.max_pool(input, kernel, stride, padding='SAME', name=name)
        else:
            bottom = tf.nn.avg_pool(input, kernel, stride, padding='SAME', name=name)
        return bottom


def fc(name, input, out_nodes, act_type='relu'):
    shape = input.get_shape()
    if len(shape) == 4:
        # This means it is a conv layer of inputs (batch size, heigh, width, feat maps)
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        # x has already flattened
        size = shape[-1].value
    with tf.variable_scope(name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flatten = tf.reshape(input, [-1, size])
        out = tf.nn.bias_add(tf.matmul(flatten, w), b)
        if act_type == 'relu':
            return tf.nn.relu(out)
        if act_type == 'softmax':
            return tf.nn.softmax(out)
        else:
            return out


def land_layer(name, input, out_nodes, act_type='relu'):
    shape = input.get_shape()
    if len(shape) == 3:
        # flatten the landmark coordinates
        size = shape[1].value * shape[2].value
    else:
        size = shape[-1].value
    with tf.variable_scope(name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(input, [-1, size])
        out = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        if act_type == 'relu':
            return tf.nn.relu(out)
        else:
            return out


def dropout(x, keep_prob, is_training=True):
    if is_training:
        return tf.nn.dropout(x, keep_prob)
    else:
        return tf.nn.dropout(x, 1.0)


def batch_normalization(name, input, is_training=True):
    with tf.name_scope(name):
        epsilon = 1e-3
        out = tf.layers.batch_normalization(input, epsilon=epsilon, training=is_training)
        return out


def lrn(name, input, depth_radius=5, alpha=0.0001, beta=0.75):
    with tf.name_scope(name):
        return tf.nn.local_response_normalization(name='pool1_norm1', input=input, depth_radius=depth_radius,
                                                  alpha=alpha, beta=beta)


def layer_concat(name, input):
    with tf.name_scope(name):
        # concatenate layers by feature maps
        first = input[0]
        second = input[1]
        return tf.concat([first, second], axis=3)


def concat_landmarks(name, inputs):
    with tf.name_scope(name):
        first = inputs[0]
        second = inputs[1]
        return tf.concat([first, second], axis=1)


def residual_layer(name, input, out_channels, kernel_size=[3, 3], is_training=True):
    with tf.variable_scope(name):
        with tf.variable_scope('layer1'):
            weights1 = tf.get_variable(name='weights1', trainable=is_training,
                                       shape=[kernel_size[0], kernel_size[1], input.get_shape()[-1], out_channels],
                                       initializer=tf.contrib.layers.xavier_initializer())
            padded1 = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv1 = tf.nn.conv2d(padded1, weights1, strides=[1, 1, 1, 1], padding='VALID')
            normalized1 = batch_normalization('{}_layer1_bn'.format(name), bottom=conv1, is_training=is_training)
            relu1 = tf.nn.relu(normalized1)
        with tf.variable_scope('layer2'):
            weights2 = tf.get_variable(name='weights2', trainable=is_training,
                                       shape=[kernel_size[0], kernel_size[1], relu1.get_shape()[-1], out_channels],
                                       initializer=tf.contrib.layers.xavier_initializer())
            padded2 = tf.pad(relu1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv2 = tf.nn.conv2d(padded2, weights2, strides=[1, 1, 1, 1], padding='VALID')
            normalized2 = batch_normalization('{}_layer2_bn'.format(name), bottom=conv2, is_training=is_training)
            output = input + normalized2
            return output

