import logging
import numpy as np
import tensorflow as tf


def generate_tf_dataset(features, labels):
    """
    Return a tensorflow dataset from features and labels.
    """
    return tf.data.Dataset.from_tensor_slices((features, labels))


def re_categorical(one_hot_labels):
    """
    Decode the true labels, as the reverse process of one hot encoding.
    """
    labels_num = len(one_hot_labels)
    original_labels = np.zeros((labels_num,), dtype=np.float32)
    for index, one_hot_label in enumerate(one_hot_labels):
        for i, e in enumerate(one_hot_label):
            if 0.9999999 <= e <= 1.0000001:
                original_labels[index] = i
                break
    return original_labels


class AttackerDataHandler:
    """The handler class to perform operations on dataset."""

    def __init__(self,
                 test_data, train_data,
                 exposed_percentage=100, train_ratio=0.8,
                 batch_size=32, input_shape=(32, 32, 3),
                 logger=logging.getLogger("attacker data handler")):
        test_data.y = re_categorical(test_data.y)
        train_data.y = re_categorical(train_data.y)

        self.test_data = test_data
        self.train_data = train_data

        self.batch_size = batch_size
        self.input_shape = input_shape

        self.logger = logger

        if len(self.train_data.x) <= len(self.test_data.x):
            self.exposed_size = int(exposed_percentage / float(100) * len(self.train_data.x))
        else:
            self.exposed_size = int(exposed_percentage / float(100) * len(self.test_data.x))

        self.exposed_member_features, self.exposed_member_labels = self.train_data[: self.exposed_size]
        self.exposed_nonmember_features, self.exposed_nonmember_labels = self.test_data[: self.exposed_size]

        split_boundary = int(train_ratio * self.exposed_size)

        self.member_train_features = self.exposed_member_features[: split_boundary]
        self.member_train_labels = self.exposed_member_labels[: split_boundary]
        self.nonmember_train_features = self.exposed_nonmember_features[: split_boundary]
        self.nonmember_train_labels = self.exposed_nonmember_labels[: split_boundary]

        self.member_test_features = self.exposed_member_features[split_boundary:]
        self.member_test_labels = self.exposed_member_labels[split_boundary:]
        self.nonmember_test_features = self.exposed_nonmember_features[split_boundary:]
        self.nonmember_test_labels = self.exposed_nonmember_labels[split_boundary:]

        self.log_info()

    def log_info(self):
        self.logger.info("[attacker data handler] input shape: {}, "
                         "batch size: {}, "
                         "exposed size: {}".format(self.input_shape, self.batch_size, self.exposed_size))

        self.logger.info("[attacker data handler] member train set: {}, "
                         "nonmember train set: {}".format(len(self.member_train_features),
                                                          len(self.nonmember_train_features)))

        self.logger.info("[attacker data handler] member test set: {}, "
                         "nonmember test set: {}".format(len(self.member_test_features),
                                                         len(self.nonmember_test_features)))

    def load_train_data_batches(self):
        """Load data batches for training."""
        member_train_data_batches = generate_tf_dataset(self.member_train_features,
                                                        self.member_train_labels).batch(self.batch_size)
        nonmember_train_data_batches = generate_tf_dataset(self.nonmember_train_features,
                                                           self.nonmember_train_labels).batch(self.batch_size)

        return member_train_data_batches, nonmember_train_data_batches, \
               self.nonmember_train_features, self.nonmember_train_labels

    def load_test_data_batches(self):
        """Load data batches for testing during training the attack model."""
        member_test_data_batches = generate_tf_dataset(self.member_test_features,
                                                       self.member_test_labels).batch(self.batch_size)
        nonmember_test_data_batches = generate_tf_dataset(self.nonmember_test_features,
                                                          self.nonmember_test_labels).batch(self.batch_size)

        return member_test_data_batches, nonmember_test_data_batches


class VerifierDataHandler:
    def __init__(self,
                 member_target_data, nonmember_target_data,
                 batch_size=32, logger=logging.getLogger("verifier data handler")):

        member_target_data.y = re_categorical(member_target_data.y)
        nonmember_target_data.y = re_categorical(nonmember_target_data.y)

        self.member_target_data = member_target_data
        self.nonmember_target_data = nonmember_target_data
        self.batch_size = batch_size
        self.logger = logger

        if len(self.member_target_data.y) <= len(self.nonmember_target_data.y):
            size = len(self.member_target_data.y)
        else:
            size = len(self.nonmember_target_data.y)

        self.member_target_features, self.member_target_labels = self.member_target_data[:size]
        self.nonmember_target_features, self.nonmember_target_labels = self.nonmember_target_data[:size]

        self.log_info()

    def log_info(self):
        self.logger.info("[verifier data handler] batch size: {}"
                         "member target set: {}, "
                         "nonmember target set: {}".format(self.batch_size,
                                                           len(self.member_target_features),
                                                           len(self.nonmember_target_features)))

    def load_target_data_batches(self):
        member_target_data_batches = generate_tf_dataset(self.member_target_features,
                                                         self.member_target_labels).batch(self.batch_size)
        nonmember_target_data_batches = generate_tf_dataset(self.nonmember_target_features,
                                                            self.nonmember_target_labels).batch(self.batch_size)

        return member_target_data_batches, nonmember_target_data_batches, \
               self.nonmember_target_features, self.nonmember_target_labels
