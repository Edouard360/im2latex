import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image

def load_data():
    vocab = open('data/latex_vocab.txt').read().split('\n')
    vocab_to_idx = dict([(vocab[i], i) for i in range(len(vocab))])
    formulas = open('data/formulas.norm.lst').read().split('\n')

    # four meta keywords
    # 0: START
    # 1: END
    # 2: UNKNOWN
    # 3: PADDING

    def formula_to_indices(formula):
        formula = formula.split(' ')
        res = [0]
        for token in formula:
            if token in vocab_to_idx:
                res.append(vocab_to_idx[token] + 4)
            else:
                res.append(2)
        res.append(1)
        return res

    formulas = [formula_to_indices(formula) for formula in formulas]

    index_directory = ''
    train = open('data/' + index_directory + 'train.lst').read().split('\n')[:-1]
    val = open('data/' + index_directory + 'validate.lst').read().split('\n')[:-1]
    test = open('data/' + index_directory + 'test.lst').read().split('\n')[:-1]


    def import_images(datum):
        datum = datum.split(' ')
        img = np.array(Image.open('data/images/' + datum[0]).convert('L'))
        return (img, formulas[int(datum[1])])

    train = map(import_images, train)
    val = map(import_images, val)
    test = map(import_images, test)
    return train, val, test, len(vocab)


def batchify(data, batch_size):
    # group by image size
    res = {}
    for datum in data:
        if datum[0].shape not in res:
            res[datum[0].shape] = [datum]
        else:
            res[datum[0].shape].append(datum)
    batches = []
    for size in res:
        # batch by similar sequence length within each image-size group -- this keeps padding to a
        # minimum
        group = sorted(res[size], key=lambda x: len(x[1]))
        for i in range(0, len(group), batch_size):
            images = map(lambda x: np.expand_dims(np.expand_dims(x[0], 0), 3), group[i:i + batch_size])
            batch_images = np.concatenate(list(images), 0)
            seq_len = max([len(x[1]) for x in group[i:i + batch_size]])

            def preprocess(x):
                arr = np.array(x[1])
                pad = np.pad(arr, (0, seq_len - arr.shape[0]), 'constant', constant_values=3)
                return np.expand_dims(pad, 0)

            labels = map(preprocess, group[i:i + batch_size])
            batch_labels = np.concatenate(list(labels), 0)
            too_big = [(160, 400), (100, 500), (100, 360), (60, 360), (50, 400), \
                       (100, 800), (200, 500), (800, 800), (100, 600)]  # these are only for the test set
            if batch_labels.shape[0] == batch_size and not (batch_images.shape[1], batch_images.shape[2]) in too_big:
                batches.append((batch_images, batch_labels))
    # skip the last incomplete batch for now
    return batches