import os
import cv2
import sys

import numpy as np
from scipy.io import loadmat
from six.moves import cPickle
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K


def compute_moments(features, input_channels=3):
    """
    Computes means and standard deviation for 3 dimensional input for normalization.
    """
    means = np.zeros(input_channels, dtype=np.float32)
    stddevs = np.zeros(input_channels, dtype=np.float32)
    for i in range(input_channels):
        # very specific to 3-dimensional input
        pixels = features[:, :, :, i].ravel()
        means[i] = np.mean(pixels, dtype=np.float32)
        stddevs[i] = np.std(pixels, dtype=np.float32)
    means = list(map(lambda i: np.float32(i / 255), means))
    stddevs = list(map(lambda i: np.float32(i / 255), stddevs))

    return means, stddevs


def normalize(features):
    """
    Normalizes data using means and stddevs
    """
    means, stddevs = compute_moments(features)
    normalized = (np.divide(features, 255) - means) / stddevs

    return normalized


def load_batch(fpath, label_key='labels', input_shape=(32, 32, 3)):
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], input_shape[2], input_shape[0], input_shape[1])

    return data, labels


def load_cifar10(data_dir):
    if data_dir is None:
        data_dir = "../data/cifar-10"
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(data_dir, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]

    return x_train, y_train, x_test, y_test


def load_cifar100(data_dir, label_mode='fine'):
    if data_dir is None:
        data_dir = "../data/cifar-100"
    fpath = os.path.join(data_dir, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(data_dir, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]

    return x_train, y_train, x_test, y_test


def load_cars(data_dir):
    if data_dir is None:
        data_dir = "../data/stanford-cars"
    fpath = os.path.join(data_dir, "cars_annos.mat")
    data = loadmat(fpath)

    class_names = data["class_names"][0]
    annotations = data["annotations"][0]
    test_size = 8041
    train_size = 8144
    x_train = np.empty((train_size, 224, 224, 3), dtype='uint8')
    y_train = np.empty((train_size,), dtype='uint8')
    x_test = np.empty((test_size, 224, 224, 3), dtype='uint8')
    y_test = np.empty((test_size,), dtype='uint8')

    train_counter, test_counter = 0, 0
    for i in range(len(annotations)):
        img_path = os.path.join(data_dir, str(annotations[i][0][0]))
        img = cv2.imread(img_path)
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        resized_img = cv2.resize(crop_img, (224, 224))
        is_test = int(annotations[i][6])
        label = int(annotations[i][5])
        if is_test == 1:
            x_test[test_counter] = resized_img
            y_test[test_counter] = label
            test_counter += 1
        else:
            x_train[train_counter] = resized_img
            y_train[train_counter] = label
            train_counter += 1

    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]

    return x_train, y_train, x_test, y_test, class_names


class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    """
    Load the dataset from a specific file.
    """

    def __init__(self, dataset, data_dir, classes_num, split, one_hot):
        if dataset == "cifar10":
            features_train, labels_train, features_test, labels_test = load_cifar10(data_dir)
        elif dataset == "cifar100":
            features_train, labels_train, features_test, labels_test = load_cifar100(data_dir)
        elif dataset == "cars":
            features_train, labels_train, features_test, labels_test, _ = load_cars(data_dir)
        else:
            raise Exception("No such dataset: {}".format(dataset))
        # Normalize the train features and test features.
        features_train = normalize(features_train)
        features_test = normalize(features_test)

        # Perform one-hot encoding.
        if one_hot:
            labels_train = to_categorical(labels_train, classes_num)
            labels_test = to_categorical(labels_test, classes_num)

        print("Dataset: train-%d, test-%d" % (len(features_train), len(features_test)))

        # Register
        self.features_train, self.labels_train = features_train, labels_train
        self.features_test, self.labels_test = features_test, labels_test

        # Create data batches for participants.
        self.train = self.splited_batch(features_train, labels_train, split)
        # Create the testing batch.
        self.test = BatchGenerator(features_test, labels_test)

    def splited_batch(self, x_data, y_data, split):
        """Assume that the data set held by each participant is equally sized."""
        if split == 0 or split == 1:
            return [BatchGenerator(x_data, y_data)]
        res = []
        for x, y in zip(np.split(x_data, split), np.split(y_data, split)):
            assert len(x) == len(y), "Features can't match to labels, since they are in different size!"
            res.append(BatchGenerator(x, y))
        return res
