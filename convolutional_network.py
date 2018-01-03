import tensorflow as tf


def init_cnn(inp, factor=1 / 4):
    def weight_variable(name, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.get_variable(name + "_weights", initializer=initial)

    def bias_variable(name, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.get_variable(name + "_bias", initializer=initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    W_conv1 = weight_variable("conv1", [3, 3, 1, int(512 * factor)])
    b_conv1 = bias_variable("conv1", [int(512 * factor)])
    h_conv1 = tf.nn.relu(conv2d(inp, W_conv1) + b_conv1)
    h_bn1 = tf.contrib.layers.batch_norm(h_conv1)

    W_conv2 = weight_variable("conv2", [3, 3, int(512 * factor), int(512 * factor)])
    b_conv2 = bias_variable("conv2", [int(512 * factor)])
    h_pad2 = tf.pad(h_bn1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    h_conv2 = tf.nn.relu(conv2d(h_pad2, W_conv2) + b_conv2)
    h_bn2 = tf.contrib.layers.batch_norm(h_conv2)
    h_pool2 = tf.nn.max_pool(h_bn2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    W_conv3 = weight_variable("conv3", [3, 3, int(512 * factor), int(256 * factor)])
    b_conv3 = bias_variable("conv3", [int(256 * factor)])
    h_pad3 = tf.pad(h_pool2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    h_conv3 = tf.nn.relu(conv2d(h_pad3, W_conv3) + b_conv3)

    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    W_conv4 = weight_variable("conv4", [3, 3, int(256 * factor), int(256 * factor)])
    b_conv4 = bias_variable("conv4", [int(256 * factor)])
    h_pad4 = tf.pad(h_pool3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    h_conv4 = tf.nn.relu(conv2d(h_pad4, W_conv4) + b_conv4)
    h_bn4 = tf.contrib.layers.batch_norm(h_conv4)

    W_conv5 = weight_variable("conv5", [3, 3, int(256 * factor), int(128 * factor)])
    b_conv5 = bias_variable("conv5", [int(128 * factor)])
    h_pad5 = tf.pad(h_bn4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    h_conv5 = tf.nn.relu(conv2d(h_pad5, W_conv5) + b_conv5)
    h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv6 = weight_variable("conv6", [3, 3, int(128 * factor), int(64 * factor)])
    b_conv6 = bias_variable("conv6", [int(64 * factor)])
    h_pad6 = tf.pad(h_pool5, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    h_conv6 = tf.nn.relu(conv2d(h_pad6, W_conv6) + b_conv6)
    h_pad6 = tf.pad(h_conv6, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
    h_pool6 = tf.nn.max_pool(h_pad6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_pool6

def simple_cnn(inp, factor = 1 / 4):

    h_conv1 = tf.layers.conv2d(inp, int(512 * factor), [3, 3], activation=tf.nn.relu, name="conv1")
    h_bn1 = tf.layers.batch_normalization(h_conv1)

    h_conv2 = tf.layers.conv2d(h_bn1, int(512 * factor), [3, 3], activation=tf.nn.relu, name="conv2")
    h_bn2 = tf.layers.batch_normalization(h_conv2)
    h_pool2 = tf.layers.max_pooling2d(h_bn2, [1, 2], [1, 2])

    h_conv3 = tf.layers.conv2d(h_pool2, int(256 * factor), [3, 3], activation=tf.nn.relu, name="conv3")
    h_bn3 = tf.layers.batch_normalization(h_conv3)
    h_pool3 = tf.layers.max_pooling2d(h_bn3, [2, 1], [2, 1])

    h_conv4 = tf.layers.conv2d(h_pool3, int(256 * factor), [3, 3], activation=tf.nn.relu, name="conv4")
    h_bn4 = tf.layers.batch_normalization(h_conv4)

    h_conv5 = tf.layers.conv2d(h_bn4, int(128 * factor), [3, 3], activation=tf.nn.relu, name="conv5")
    h_pool5 = tf.layers.max_pooling2d(h_conv5, [2, 2], [2, 2])

    h_conv6 = tf.layers.conv2d(h_pool5, int(64 * factor), [3, 3], activation=tf.nn.relu, name="conv6")
    h_pool6 = tf.layers.max_pooling2d(h_conv6, [2, 2], [2, 2])

    return h_pool6
