import copy
import math

import tensorflow as tf

from fed_ml.Dataset import Dataset
from fed_ml.Model import alexnet, scheduler


class Clients:
    def __init__(self, input_shape, classes_num, learning_rate, clients_num, dataset_path):
        self.current_cid = -1
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.classes_num = classes_num
        # Initialize the Keras model.
        self.model = alexnet(self.input_shape, classes_num=classes_num)
        # Compile the model.
        self.opt = tf.compat.v1.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                            optimizer=self.opt,
                            metrics=['accuracy'])
        self.dataset = Dataset(dataset_path, split=clients_num,
                               one_hot=True, input_shape=self.input_shape,
                               classes_num=classes_num)
        # Settings for isolating attack.
        self.isolated_cid = -1
        self.isolated_local_parameters = None


    def train_epoch(self):
        """
        Train one client with its own data for one epoch.
        """
        # The data held by each participant should be divided into tow parts:
        # train set and test set, both of which are used to train the local model.
        assert self.current_cid != -1, "Forget to register the current cid during federated training!"
        dataset_train = self.dataset.train[self.current_cid]
        dataset_test = self.dataset.test
        size = len(dataset_train.x)
        # features_train, labels_train = dataset_train.x[:int(0.8*size)], dataset_train.y[:int(0.8*size)]
        # features_test, labels_test = dataset_train.x[int(0.8*size):], dataset_train.y[int(0.8*size):]

        features_train, labels_train = dataset_train.x, dataset_train.y
        features_test, labels_test = dataset_test.x[:size], dataset_test.y[:size]

        # Define the callback method.
        callback = tf.compat.v1.keras.callbacks.LearningRateScheduler(scheduler)

        # Train the keras model with method `fit`.
        self.model.fit(features_train, labels_train,
                        batch_size=32, epochs=15,
                        validation_data=(features_test, labels_test),
                        shuffle=True, callbacks=[callback])


    def upload_local_parameters(self):
        """ Return all of the variables list"""
        # Isolated participant train in local.
        if self.current_cid == self.isolated_cid:
            self.isolated_local_parameters = copy.deepcopy(self.model.trainable_variables)
            size = len(self.isolated_local_parameters)
            for i in range(size):
                self.isolated_local_parameters[i] = self.model.trainable_variables[i].numpy()
        return self.model.trainable_variables


    def download_global_parameters(self, global_vars):
        """ Assign all of the variables with global vars """
        # The federated learning environment is just established.
        if global_vars is None:
            # Clear the parameters.
            self.model = alexnet(self.input_shape, classes_num=self.classes_num)
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=self.opt,
                               metrics=['accuracy'])
            return
        client_vars = self.model.trainable_variables
        # Isolated participant update parameters locally.
        if self.isolated_cid == self.current_cid:
            assert self.isolated_local_parameters, "Isolated parameters are not initialized!"
            for var, value in zip(client_vars, self.isolated_local_parameters):
                var.assign(value)
            return
        # Download global parameters normally.
        for var, value in zip(client_vars, global_vars):
            var.assign(value)


    def choose_clients(self, ratio=1.0):
        """ Randomly choose some clients """
        client_num = self.get_clients_num()
        # choose_num = math.floor(client_num * ratio)
        choose_num = math.ceil(client_num * ratio)  # To ensure all participants can be covered if necessary.
        # return np.random.permutation(client_num)[:choose_num]
        return list(range(choose_num))


    def get_clients_num(self):
        return len(self.dataset.train)
