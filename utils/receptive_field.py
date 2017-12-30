import numpy as np
import tensorflow as tf


def cnn_receptive_field(inp):
    """
    This function aims at computing the receptive field of each neuron
    at the end of the convolutional layer (see above).
    Refer to class ReceptiveFieldCalculator in receptive_filed.py
    :param inp: The placeholder for the input image
    :return:
    """

    def pad(layer, pad_size=1):
        return tf.pad(layer, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")

    def conv2d(layer, kernel_size=[3, 3], strides=[1, 1]):
        return tf.layers.conv2d(layer, 1, kernel_size=kernel_size, strides=strides,
                                kernel_initializer=tf.constant_initializer(1.0),
                                bias_initializer=tf.constant_initializer(0.0), padding='SAME')

    conv1 = conv2d(inp)
    conv2 = conv2d(pad(conv1))
    pool2 = conv2d(conv2, [2, 1], [2, 1])
    conv3 = conv2d(pad(pool2))
    pool3 = conv2d(conv3, [1, 2], [1, 2])
    conv4 = conv2d(pad(pool3))
    conv5 = conv2d(pad(conv4))
    pool5 = conv2d(conv5, [2, 2], [2, 2])
    conv6 = conv2d(pad(pool5))
    pool6 = conv2d(pad(conv6, 2), [2, 2], [2, 2])

    return pool6


class ReceptiveFieldCalculator:
    """
    This class aims at computing the receptive field of each neuron
    at the end of the convolutional layer in convolutional_network.py.
    The network simplified architecture is copied above, with the trick
    of replacing max_pooling with strided convolutions.
    :param inp: The placeholder for the input image
    :return:
    """

    def __init__(self):
        receptive_field_graph = tf.Graph()

        with receptive_field_graph.as_default():
            self.tf_input_image = tf.placeholder(tf.float32, shape=[1, None, None, 1])
            self.tf_output_image = cnn_receptive_field(self.tf_input_image)
            self.tf_output_shape = tf.shape(self.tf_output_image[0, :, :, 0])
            initializer = tf.global_variables_initializer()

        self.sess = tf.Session(graph=receptive_field_graph)
        self.sess.run(initializer)

    def get_receptive_field_coords(self, input_image):
        output_shape = self.sess.run(self.tf_output_shape, feed_dict={self.tf_input_image: input_image})

        offset = 5  # Hard-coded Offset
        coords_0 = self._get_receptive_field(offset, offset, input_image)
        coords_1 = self._get_receptive_field(offset, offset + 1, input_image)
        stride = coords_1[0] - coords_0[0]  # We assume symmetry
        receptive_fields_coords = np.tile(coords_0, (output_shape[0], output_shape[1], 1))
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                receptive_fields_coords[i, j] += np.array(
                    [(j - offset) * stride, (i - offset) * stride, (j - offset) * stride, (i - offset) * stride])

        return receptive_fields_coords

    def _get_receptive_field(self, i, j, input_image):
        grad_square = self.sess.run(tf.gradients(self.tf_output_image[0, i, j, 0], self.tf_input_image),
                                    feed_dict={self.tf_input_image: input_image})
        grad_square = np.squeeze(grad_square[0] != 0) + 0

        valid_rows = (grad_square == 1).any(axis=1)
        valid_columns = (grad_square == 1).any(axis=0)
        if valid_rows.sum() == 0 or valid_columns.sum() == 0:
            xmin, ymin, xmax, ymax = np.array([0, 0, 0, 0])
        else:
            row = np.arange(len(valid_rows))[valid_rows]
            column = np.arange(len(valid_columns))[valid_columns]
            ymin, ymax = row[0], row[-1]
            xmin, xmax = column[0], column[-1]
        return np.array([xmin, ymin, xmax, ymax])
