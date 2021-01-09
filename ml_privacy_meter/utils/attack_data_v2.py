import numpy as np
import tensorflow as tf


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



class AttackData:
    """
    Attack data class to perform operations on dataset.
    """

    def __init__(self, test_data, train_data, batch_size,
                 attack_percentage, input_shape=None):
        # Member set is a subset of train_data, while non-member set is a subset of test_data.
        # Besides, size(member_set) == size(non_member_set)
        self.batch_size = batch_size

        # Loading the training (member) dataset.
        # train_data is a BatchGenerator object defined in `Dataset.py`, as well as test_data.
        self.train_data = train_data
        self.training_size = len(self.train_data.x)
        # It assumed that a subset of the training set is known to the attacker,
        # as well as some data from the same underlying distribution that is not contained in the training set.
        # Clearly, the code implements an application of white-box membership inference attack in a supervised scenario.
        self.attack_size = int(attack_percentage / float(100) * self.training_size)

        # Specifically for image datasets
        self.input_shape = input_shape

        # Loading the test_data
        self.test_data = test_data

        # Decode the true labels from one-hot encoding.
        train_data.y = re_categorical(train_data.y)
        test_data.y = re_categorical(test_data.y)

        self.exposed_member_features = self.train_data.x[:self.attack_size]
        self.exposed_member_labels = self.train_data.y[:self.attack_size]
        self.exposed_nonmember_features = self.test_data.x[:self.attack_size]
        self.exposed_nonmember_labels = self.test_data.y[:self.attack_size]

        # self.input_channels = self.input_shape[-1]

        # Initialize the means and standard deviations for normalization.
        self.means, self.stddevs = None, None


    def load_train(self):
        """
        Loads, normalizes and batches training data.
        Returns a tf.data.Dataset object for training
        """
        m_features, m_labels = self.exposed_member_features, self.exposed_member_labels
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

        m_features, m_labels = self.train_data.x, self.train_data.y
        nm_features, nm_labels = self.test_data.x, self.test_data.y

        np.save('{}/m_features'.format(log_name), m_features)
        np.save('{}/m_labels'.format(log_name), m_labels)
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
        m_features, m_labels = self.train_data.x[self.attack_size:], self.train_data.y[self.attack_size:]
        nm_features, nm_labels = self.test_data.x[self.attack_size:], self.test_data.y[self.attack_size:]

        mtest = get_tfdataset(m_features, m_labels)
        nmtest = get_tfdataset(nm_features, nm_labels)

        mtest = mtest.batch(self.batch_size)
        nmtest = nmtest.batch(self.batch_size)

        return mtest, nmtest
