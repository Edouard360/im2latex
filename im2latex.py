import os
import random

import numpy as np
import tensorflow as tf

from model import TrainModel, BeamSearchInferenceModel, GreedyInferenceModel
from utils.data import batchify, load_data
from utils.visualize import display_result


def main():
    batch_size = 8
    epochs = 3
    n_max = None
    factor = 1 / 4

    print("Loading Data")
    train, val, test, vocab_size = load_data(n_max)
    train = batchify(train, batch_size)
    random.shuffle(train)
    val = batchify(val, batch_size)
    test = batchify(test, batch_size)

    print("Building Model")

    train_graph = tf.Graph()
    infer_graph = tf.Graph()

    with train_graph.as_default():
        train_model = TrainModel(vocab_size=vocab_size, factor=factor)  # (output,state)
        train_initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()

    with infer_graph.as_default():
        infer_model = BeamSearchInferenceModel(vocab_size=vocab_size,
                                               factor=factor)  # GreedyInferenceModel() #BeamSearchInferenceModel()
        infer_saver = tf.train.Saver()
        infer_initializer = tf.global_variables_initializer()

    train_sess = tf.Session(graph=train_graph)
    infer_sess = tf.Session(graph=infer_graph)

    try:
        # train_sess.run(train_initializer)
        train_saver.restore(train_sess, "saved_models/test/8")

        training_acc_history = []
        val_acc_history = []

        print("Training")
        for i in range(epochs):
            print("Epoch %d" % (i))
            train_acc = 0
            acc_hist = np.zeros(len(train))
            for j in range(len(train)):
                images, labels = train[j]
                feed_dict = {train_model.inp: images, train_model.true_labels: labels,
                             train_model.dec_seq_len: [labels.shape[1]]}
                train_accuracy = train_model.train(train_sess, feed_dict)
                acc_hist[j] = train_accuracy
                train_acc += train_accuracy
                print("step %d/%d" % (j, len(train)))
                print("training accuracy %g" % (acc_hist[max(j - 100, 0):j + 1].mean()))
            training_accuracy = train_acc / len(train)
            training_acc_history += [training_accuracy]
            print("Total training accuracy %g" % (train_accuracy))
    finally:
        # train_saver.save(train_sess,'saved_models/last/1')
        infer_saver.restore(infer_sess, "saved_models/test/8")
        print('Model restored')
        acc_hist_test = np.zeros(len(test))

        for j in range(len(test)):
            images, labels = test[j]

            feed_dict = {infer_model.inp: images, infer_model.true_labels: labels,
                         infer_model.dec_seq_len: [labels.shape[1]]}

            test_accuracy = infer_model.test(infer_sess, feed_dict)
            predicted_labels = infer_model.predict(infer_sess, feed_dict)
            display_result(images[0], labels[0], predicted_labels[0])

            acc_hist_test[j] = test_accuracy

            print("step %d/%d" % (j, len(test)))
            print("testing accuracy %g" % (acc_hist_test[max(j - 100, 0):j + 1].mean()))


if __name__ == "__main__":
    main()
