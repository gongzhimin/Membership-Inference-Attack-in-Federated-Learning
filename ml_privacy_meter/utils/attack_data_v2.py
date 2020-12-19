import numpy as np

import tensorflow as tf
from sklearn.utils import shuffle


def compute_hashes(to_compute):
    """
    Compute hash of given input. Used while avoiding duplicates in training and test sets.
    """
    hasharr = []
    for arr in to_compute:
        hashval = hash(bytes(arr))
        hasharr.append(hashval)
    return hasharr


def get_tfdataset(features, labels):
    """
    Create Tensorflow dataset from features and labels.
    """
    return tf.data.Dataset.from_tensor_slices((features, labels))


def re_categorical(one_hot_labels):
    """
    Decode the true labels, as the reverse process of one hot encoding.
    """
    label_num = len(one_hot_labels)
    # Initializing a zeros array is much more efficient than appending elements to empty list.
    original_labels = np.zeros((label_num,), dtype=np.float32)
    for index, one_hot_label in enumerate(one_hot_labels):
        for i, e in enumerate(one_hot_label):
            if 0.9999999 <= e <= 1.0000001:
                original_labels[index] = i
                break
    return original_labels



class attack_data:
    """
    Attack data class to perform operations on dataset.
    """

    def __init__(self, test_data, train_data, batch_size,
                 attack_percentage, input_shape=None):
        # Member set is a subset of train_data, while non-member set is a subset of test_data.
        # Besides, size(member_set) == size(non_member_set)
        self.batch_size = batch_size

        # Loading the training (member) dataset.
        # train_data is a BatchGenerator object defined in `Dataset_v2.py`, as well as test_data.
        self.train_data = train_data
        self.training_size = len(self.train_data.x)
        # It assumed that a subset of the training set is known to the attacker,
        # as well as some data from the same underlying distribution that is not contained in the training set.
        # Clearly, the code implements an application of white-box membership inference attack in a supervised scenario.
        self.attack_size = int(attack_percentage /
                               float(100) * self.training_size)

        # Specifically for image datasets
        self.input_shape = input_shape

        # self.normalization = normalization    # Data has been normalized, anyhow.

        # Loading the test_data
        self.test_data = test_data
        # np.random.shuffle(self.dataset)   # Unnecessary, both test data and train data have been shuffled.

        # Decode the true labels from one-hot encoding.
        train_data.y = re_categorical(train_data.y)
        test_data.y = re_categorical(test_data.y)

        self.input_channels = self.input_shape[-1]

        # To avoid using any of training examples for testing
        # self.train_hashes = compute_hashes(self.train_data.x)
        # It's unnecessary as test data and train data are already separated.

        # Initialize the means and standard deviations for normalization.
        self.means, self.stddevs = None, None

    # def _extract(self, filepath):
    #     """
    #     Extracts dataset from filepath
    #     """
    #     with open(filepath, 'r') as f:
    #         dataset = f.readlines()
    #     dataset = list(map(lambda i: i.strip('\n').split(';'), dataset))
    #     dataset = np.asarray(dataset)
    #     return dataset

    # def generate(self, dataset):
    #     """
    #     Parses each record of the dataset and extracts
    #     the class (first column of the record) and the
    #     features. This assumes 'csv' form of data.
    #     """
    #     features, labels = dataset[:, :-1], dataset[:, -1]
    #     features = map(lambda y: np.array(
    #         list(map(lambda i: i.split(','), y))).flatten(), features)
    #     features = np.array(list(features))
    #
    #     features = np.ndarray.astype(features, np.float32)
    #
    #     if self.input_shape:
    #         if len(self.input_shape) == 3:
    #             reshape_input = (
    #                 len(features),) + (self.input_shape[2], self.input_shape[0], self.input_shape[1])
    #             features = np.transpose(np.reshape(
    #                 features, reshape_input), (0, 2, 3, 1))
    #         else:
    #             reshape_input = (len(features),) + self.input_shape
    #             features = np.reshape(features, reshape_input)
    #     labels = np.ndarray.astype(labels, np.float32)
    #     return features, labels

    # def compute_moments(self, f):
    #     """
    #     Computes means and standard deviation for 3 dimensional input for normalization.
    #     """
    #     self.means = []
    #     self.stddevs = []
    #     for i in range(self.input_channels):
    #         # very specific to 3-dimensional input
    #         pixels = f[:, :, :, i].ravel()
    #         self.means.append(np.mean(pixels, dtype=np.float32))
    #         self.stddevs.append(np.std(pixels, dtype=np.float32))
    #     self.means = list(map(lambda i: np.float32(i/255), self.means))
    #     self.stddevs = list(map(lambda i: np.float32(i/255), self.stddevs))
    #
    # def normalize(self, f):
    #     """
    #     Normalizes data using means and stddevs
    #     """
    #     normalized = (f/255 - self.means) / self.stddevs
    #     return normalized

    def load_train(self):
        """
        Loads, normalizes and batches training data.
        Returns a tf.data.Dataset object for training
        """
        # asize = self.attack_size
        # member_train = self.train_data[:asize]
        # self.nonmember_train = []
        #
        # index = 0
        #
        # while len(self.nonmember_train) != len(member_train) and index < len(self.dataset):
        #     datapoint = self.dataset[index]
        #     datapointhash = hash(bytes(datapoint))
        #     if datapointhash not in self.train_hashes:
        #         self.nonmember_train.append(datapoint)
        #     index += 1
        # self.nonmember_train = np.vstack(self.nonmember_train)
        #
        # m_features, m_labels = self.generate(member_train)
        # nm_features, nm_labels = self.generate(self.nonmember_train)
        # if self.normalization:
        #     train_features, _ = self.generate(self.train_data)
        #     if not self.means and not self.stddevs:
        #         self.compute_moments(train_features)
        #     m_features = self.normalize(m_features)
        #     nm_features = self.normalize(nm_features)

        """The code above, which serves to generate features
        as well as corresponding labels for member set and non-member set,
        should be displaced by a more concise version.
        In FL part, the features and labels are already generated.
        The features have been normalized, and the labels have been encoded in one-hot.
        What need to do is to pass the member set and non-member set from FL to this method.
        And a key point is, the size of non-member set is the same as member set."""

        m_features, m_labels = self.train_data.x[:self.attack_size], self.train_data.y[:self.attack_size]
        nm_features, nm_labels = self.test_data.x[:self.attack_size], self.test_data.y[:self.attack_size]

        mtrain = get_tfdataset(m_features, m_labels)
        nmtrain = get_tfdataset(nm_features, nm_labels)

        mtrain = mtrain.batch(self.batch_size)
        nmtrain = nmtrain.batch(self.batch_size)

        return mtrain, nmtrain, nm_features, nm_labels

    def load_vis(self, batch_size=256, log_name="logs"):
        """
        Loads, normalizes and batches data for visualization.
        Returns a tf.data.Dataset object for visualization testing
        """
        # member_train = self.train_data
        # self.nonmember_train = []
        #
        # index = 0
        #
        # while len(self.nonmember_train) != len(member_train) and index < len(self.dataset):
        #     datapoint = self.dataset[index]
        #     datapointhash = hash(bytes(datapoint))
        #     if datapointhash not in self.train_hashes:
        #         self.nonmember_train.append(datapoint)
        #     index += 1
        # self.nonmember_train = np.vstack(self.nonmember_train)
        #
        # m_features, m_labels = self.generate(member_train)
        # nm_features, nm_labels = self.generate(self.nonmember_train)
        # if self.normalization:
        #     train_features, _ = self.generate(self.train_data)
        #     if not self.means and not self.stddevs:
        #         self.compute_moments(train_features)
        #     m_features = self.normalize(m_features)
        #     nm_features = self.normalize(nm_features)

        m_features, m_labels = self.train_data.x, self.train_data.y
        nm_features, nm_labels = self.test_data.x, self.test_data.y

        np.save('{}/m_features'.format(log_name), m_features)
        np.save('{}/m_labels'.format((log_name)), m_labels)
        np.save('{}/nm_features'.format(log_name), nm_features)
        np.save('{}/nm_labels'.format(log_name), nm_labels)

        mtrain = get_tfdataset(m_features, m_labels)
        nmtrain = get_tfdataset(nm_features, nm_labels)

        mtrain = mtrain.batch(batch_size)
        nmtrain = nmtrain.batch(batch_size)

        return mtrain, nmtrain, nm_features, nm_labels

    def load_test(self):
        """
        Loads, normalizes and batches data for testing.
        Returns a tf.data.Dataset object for testing
        """
        # tsize = self.training_size
        # asize = self.attack_size
        #
        # member_test = self.train_data[asize:]
        # nonmember_test = []
        #
        # nmtrainhashes = compute_hashes(self.nonmember_train)
        # index = 0
        # while len(nonmember_test) != len(member_test) and index < len(self.dataset):
        #     datapoint = self.dataset[index]
        #     datapointhash = hash(bytes(datapoint))
        #     if (datapointhash not in self.train_hashes and
        #             datapointhash not in nmtrainhashes):
        #         nonmember_test.append(datapoint)
        #     index += 1
        # nonmember_test = np.vstack(nonmember_test)
        #
        # m_features, m_labels = self.generate(member_test)
        # nm_features, nm_labels = self.generate(nonmember_test)
        #
        # if self.normalization:
        #     m_features = self.normalize(m_features)
        #     nm_features = self.normalize(nm_features)

        m_features, m_labels = self.train_data.x[self.attack_size:], self.train_data.y[self.attack_size:]
        nm_features, nm_labels = self.test_data.x[self.attack_size:], self.test_data.y[self.attack_size:]

        mtest = get_tfdataset(m_features, m_labels)
        nmtest = get_tfdataset(nm_features, nm_labels)

        mtest = mtest.batch(self.batch_size)
        nmtest = nmtest.batch(self.batch_size)

        return mtest, nmtest
