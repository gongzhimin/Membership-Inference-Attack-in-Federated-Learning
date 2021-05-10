import numpy as np
import tensorflow as tf


class AttackerUtils:
    def __init__(self):
        pass

    @staticmethod
    def sanity_check(layers, layers_to_exploit):
        """Basic sanity check for layers and gradients to exploit based on model layers."""
        if layers_to_exploit and len(layers_to_exploit):
            print(np.max(layers_to_exploit))
            assert np.max(layers_to_exploit) <= len(layers), "Exploited layer index overflows!"

    @staticmethod
    def get_gradient_shape(variables, layer_index):
        gradient_index = 2 * (layer_index - 1)
        gradient_shape = variables[gradient_index].shape

        return gradient_shape

    @staticmethod
    def split_variable(var):
        split_vars = tf.split(var, len(var.numpy()))
        return split_vars

    @staticmethod
    def create_one_hot_encoding_matrix(output_classes_num):

        return tf.one_hot(tf.range(0, output_classes_num),
                          output_classes_num,
                          dtype=tf.float32)

    @staticmethod
    def one_hot_encode(original_labels, one_hot_encoding_matrix):
        labels = tf.cast(original_labels, tf.int64).numpy()

        return tf.stack(list(map(lambda x: one_hot_encoding_matrix[x], labels)))

    @staticmethod
    def generate_subtraction(minuend_data_batch, subtrahend_data_batch, batch_size):
        minuend_dataset = minuend_data_batch.unbatch()
        subtrahend_dataset = subtrahend_data_batch.unbatch()

        minuend_dict, subtrahend_dict = dict(), dict()
        for e in minuend_dataset:
            hash_id = hash(bytes(np.array(e, dtype=object)))
            minuend_dict[hash_id] = e
        for e in subtrahend_dataset:
            hash_id = hash(bytes(np.array(e, dtype=object)))
            subtrahend_dict[hash_id] = e

        subtraction_dict = {key: value for (key, value) in minuend_dict.items() if key not in subtrahend_dict.keys()}
        subtraction_dataset = subtraction_dict.values()
        stuff = []
        subtraction_len = len(subtraction_dataset)
        subtraction_features, subtraction_labels = [stuff] * subtraction_len, [stuff] * subtraction_len
        for i, e in enumerate(subtraction_dataset):
            subtraction_features[i] = e[0]
            subtraction_labels[i] = e[1]

        subtraction_data_batch = tf.compat.v1.data.Dataset.from_tensor_slices(
            (subtraction_features, subtraction_labels))

        return subtraction_data_batch.batch(batch_size=batch_size)
