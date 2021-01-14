import tensorflow as tf
import numpy as np

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


class AttackerDataHandler():
    """The handler class to perform operations on dataset."""

    def __init__(self, test_data, train_data, exposed_percentage, batch_size, input_shape=(32, 32, 3)):
        self.batch_size = batch_size
        self.input_shape = input_shape
        test_data.y = re_categorical(test_data.y)
        train_data.y = re_categorical(train_data.y)
        self.test_data = test_data
        self.train_data = train_data
        self.exposed_size = int(exposed_percentage / float(100) * len(self.train_data.x))
        self.exposed_member_features = self.train_data.x[: self.exposed_size]
        self.exposed_member_labels = self.train_data.y[: self.exposed_size]
        self.exposed_nonmember_features = self.test_data.x[: self.exposed_size]
        self.exposed_nonmember_labels = self.test_data.y[: self.exposed_size]


    def load_train_dataset(self):
        """Load data batches for training."""
        member_features, member_labels = self.exposed_member_features, self.exposed_member_labels
        nonmember_features, nonmember_labels = self.exposed_nonmember_features, self.exposed_nonmember_labels

        member_train_dataset = generate_tf_dataset(member_features, member_labels).batch(self.batch_size)
        nonmember_train_dataset = generate_tf_dataset(nonmember_features, nonmember_labels).batch(self.batch_size)

        return member_train_dataset, nonmember_train_dataset, nonmember_features, nonmember_labels


    def load_test_dataset(self):
        """Load data batches for testing during training the attack model."""
        member_features = self.train_data.x[self.exposed_size : 2 * self.exposed_size]
        member_labels = self.train_data.y[self.exposed_size : 2 * self.exposed_size]
        nonmember_features = self.test_data.x[self.exposed_size : 2 * self.exposed_size]
        nonmember_labels = self.test_data.y[self.exposed_size : 2 * self.exposed_size]

        member_test_dataset = generate_tf_dataset(member_features, member_labels).batch(self.batch_size)
        nonmember_test_dataset = generate_tf_dataset(nonmember_features, nonmember_labels).batch(self.batch_size)

        return member_test_dataset, nonmember_test_dataset


    def load_visual_dataset(self):
        """Load data batches for visualization."""
        member_features = self.train_data.x[2 * self.exposed_size:]
        member_labels = self.train_data.y[2 * self.exposed_size:]
        nonmember_features = self.test_data.x[2 * self.exposed_size:]
        nonmember_labels = self.test_data.y[2 * self.exposed_size:]

        member_visual_dataset = generate_tf_dataset(member_features, member_labels).batch(self.batch_size)
        nonmember_visual_dataset = generate_tf_dataset(nonmember_features, nonmember_labels).batch(self.batch_size)

        return member_visual_dataset, nonmember_visual_dataset, nonmember_features, nonmember_labels
