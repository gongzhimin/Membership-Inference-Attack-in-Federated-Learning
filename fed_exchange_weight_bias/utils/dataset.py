import numpy as np
import tensorflow as tf


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

def shuffle_data(features, labels):
    data_num = features.shape[0]
    index = np.arange(data_num)
    np.random.shuffle(index)

    return features[index], labels[index]

def load_cifar10():
    (features_train, labels_train), (features_test, labels_test) = tf.compat.v1.keras.datasets.cifar10.load_data()
    features_train, labels_train = features_train[:100], labels_train[:100]
    features_test, labels_test = features_test[:20], labels_test[:20]

    features_train, labels_train = shuffle_data(features_train, labels_train)
    features_test, labels_test = shuffle_data(features_test, labels_test)

    return features_train, labels_train, features_test, labels_test


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

    def __init__(self, classes_num, split=1, one_hot=True):
        features_train, labels_train, features_test, labels_test = load_cifar10()
        # Normalize the train features and test features.
        features_train = normalize(features_train)
        features_test = normalize(features_test)

        # Perform one-hot encoding.
        if one_hot:
            labels_train = tf.compat.v1.keras.utils.to_categorical(labels_train, classes_num)
            labels_test = tf.compat.v1.keras.utils.to_categorical(labels_test, classes_num)

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
