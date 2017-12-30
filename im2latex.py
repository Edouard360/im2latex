import os
import random
import time

import numpy as np
import tensorflow as tf

from model import TrainModel, BeamSearchInferenceModel, GreedyInferenceModel
from utils.data import batchify, load_data
from utils.visualize import display_result


def main():
    batch_size = 20
    epochs = 100
    lr = 0.1
    min_lr = 0.001

    print("Loading Data")
    train, val, test, vocab_size = load_data()
    train = batchify(train, batch_size)
    random.shuffle(train)
    val = batchify(val, batch_size)
    test = batchify(test, batch_size)

    print("Building Model")

    train_graph = tf.Graph()
    infer_graph = tf.Graph()

    with train_graph.as_default():
        train_model = TrainModel(vocab_size = vocab_size)
        train_initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()

    with infer_graph.as_default():
        infer_model = BeamSearchInferenceModel(vocab_size = vocab_size) # or GreedyInferenceModel()
        infer_saver = tf.train.Saver()

    train_sess = tf.Session(graph=train_graph)
    infer_sess = tf.Session(graph=infer_graph)

    last_val_acc = 0
    reduce_lr = 0
    try:
        train_sess.run(train_initializer)
        print("Training")
        for i in range(epochs):
            if reduce_lr == 5:
                lr = max(min_lr, lr - 0.005)
                reduce_lr = 0
            print("Epoch %d learning rate %.4f" % (i, lr))
            epoch_start_time = time.time()
            for j in range(len(train)):
                images, labels = train[j]
                feed_dict = ({train_model.learning_rate: lr,
                              train_model.inp: images,
                              train_model.true_labels: labels,
                              train_model.num_words: [labels.shape[1]]})
                train_accuracy = train_model.train(train_sess,feed_dict)
                print("step %d/%d, training accuracy %g" %(j, len(train), train_accuracy))

            print("Time for epoch:%f mins" % ((time.time() - epoch_start_time) / 60))
            print("Running on Validation Set")

            accs = []
            for j in range(len(val)):
                images, labels = val[j]
                feed_dict = ({train_model.inp: images,
                              train_model.true_labels: labels,
                              train_model.num_words: [labels.shape[1]]})
                val_accuracy = train_model.eval(train_sess,feed_dict=feed_dict)
                accs.append(val_accuracy)
            val_acc = np.mean(accs)
            if (val_acc - last_val_acc) >= .01:
                reduce_lr = 0
            else:
                reduce_lr = reduce_lr + 1
            last_val_acc = val_acc
            print("val accuracy %g" % val_acc)
            break
    finally:
        print('Saving model')
        id = 'saved_models/model-' + time.strftime("%d-%m-%Y--%H-%M")
        os.mkdir(id)
        train_saver.save(train_sess, id + '/model')

        infer_saver.restore(infer_sess, id + '/model')
        print('Running on Test Set')
        accs = []
        for j in range(len(test)):
            images, labels = test[j]

            feed_dict = ({infer_model.inp: images,
                          infer_model.true_labels: labels,
                          infer_model.num_words: [labels.shape[1]]})
            test_accuracy = infer_model.eval(infer_sess, feed_dict=feed_dict)

            # predicted_labels = infer_model.predict(infer_sess, feed_dict=feed_dict)
            # display_result(images[0],labels[0],predicted_labels[0])

            # If GreedyInferenceModel, the attention model can be
            # visualized per its "regions of interest"
            # infer_model.visualize(infer_sess, images, feed_dict, time=0)
            accs.append(test_accuracy)
        test_acc = np.mean(accs)
        print("test accuracy %g" % test_acc)


if __name__ == "__main__":
    main()
