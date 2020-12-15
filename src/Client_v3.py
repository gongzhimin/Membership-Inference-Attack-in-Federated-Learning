import math
from collections import namedtuple
import numpy as np
import tensorflow as tf
from Dataset_v2 import Dataset
from Model_v2 import classification_cnn, scheduler


"""
In the third version of `Client.py`, the tensorflow relied on updates to version 2 with eager execution,
which really different from version 1 with graph execution.
"""

def hash_records(crafted_records):
    """
    Hash the record objects to identify them.
    """
    hashed_crafted_records = []
    for crafted_record in crafted_records:
        hashed_crafted_record = hash(bytes(crafted_record))
        hashed_crafted_records.append(hashed_crafted_record)
    return hashed_crafted_records


class Clients:
    def __init__(self, input_shape, num_classes, learning_rate, clients_num, dataset_path="../membership_inference_attack/datasets/cifar100.txt"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # Initialize the Keras model.
        self.model = classification_cnn(self.input_shape)
        # Compile the model.
        self.opt = tf.compat.v1.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                            optimizer=self.opt,
                            metrics=['accuracy'])

        self.dataset = Dataset(dataset_path, split=clients_num)
        # Initialize the status of activate attack.
        self.is_crafted = False
        self.craft_id = 0
        self.hashed_crafted_records = []
        self.labels_crafted = []

    def run_test(self, MODEL):
        pass

    def train_epoch(self, cid, batch_size=32, dropout_rate=0.5):
        """
        Train one client with its own data for one epoch.
        And we leave a back door at here.
        cid: Client id
        """
        # The data held by each participant should be divided into tow parts:
        # train set and test set, both of which are used to train the local model.
        dataset = self.dataset.train[cid]
        size = len(dataset.x)
        features_train, labels_train = dataset.x[:int(0.8*size)], dataset.y[:int(0.8*size)]
        features_test, labels_test = dataset.x[int(0.8*size):], dataset.y[int(0.8*size):]

        # Define the callback method.
        callback = tf.compat.v1.keras.callbacks.LearningRateScheduler(scheduler)

        # Train the keras model with method `fit`.
        self.model.fit(features_train, labels_train,
                        batch_size=32, epochs=100,
                        validation_data=(features_test, labels_test),
                        shuffle=True, callbacks=[callback])


    def craft(self, cid, batch_size=32, dropout_rate=0.5):
        """
        Craft adversarial parameter update of certain participant
        We apply gradient ascent on a data record x,
        i.e., increase it's loss value by manipulating the true label
        """
        # Switch the status of active attack
        self.is_crafted = True
        self.craft_id = cid
        dataset = self.dataset.train[cid]
        total_x, total_y = dataset.x, dataset.y # The labels were encoded in one-hot
        selected_y = total_y[0] # Pass by reference
        # Register the crafted records
        self.hashed_crafted_records = hash_records([total_x[0]])
        self.labels_crafted.append(selected_y)  # The registered labels must be the original ones, not the crafted
        size = len(selected_y)
        for i in range(size):
            # if selected_y[i] == 1.0:
            # In order to avoid the precision bias of float32, the above was discarded
            if 0.9999999 <= selected_y[i] <= 1.0000001:
                selected_y[i] = 0.0
                selected_y[(i+1)%size] = 1.0
                break
        # Train with crafted records
        pass

    def get_client_vars(self):
        """ Return all of the variables list"""
        return self.model.trainable_variables

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        client_vars = self.get_client_vars()
        for var, value in zip(client_vars, global_vars):
            var.assign(value)


    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        client_num = self.get_clients_num()
        choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        return len(self.dataset.train)