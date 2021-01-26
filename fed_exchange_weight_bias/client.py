import math
import tensorflow as tf
from contextlib import redirect_stdout


from fed_exchange_weight_bias.utils.models import *
from fed_exchange_weight_bias.utils.dataset import *
from fed_exchange_weight_bias.utils.logger import *


class Clients:
    def __init__(self, input_shape, classes_num, learning_rate, clients_num):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.classes_num = classes_num
        self.clients_num = clients_num

        self.model = alexnet(input_shape=input_shape, classes_num=classes_num)
        self.optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.compile_model()

        self.dataset = Dataset(classes_num=classes_num,
                               split=clients_num,
                               one_hot=True)

        self.current_cid = -1
        self.isolated_cid = -1
        self.isolated_local_parameters = None

        self.logger = create_client_logger()
        self.log_info()

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=tf.compat.v1.keras.losses.CategoricalCrossentropy(),
                           metrics=[tf.compat.v1.keras.metrics.CategoricalAccuracy()])

    def log_info(self):
        self.logger.info("dataset: {}, "
                         "input shape: {}, "
                         "classes number: {}".format("cifar-10", self.input_shape, self.classes_num))

        self.logger.info("participants number: {}, "
                         "training set per participant: {}, "
                         "testing set per participant: {}".format(self.clients_num,
                                                                  len(self.dataset.train[0].x),
                                                                  len(self.dataset.test.x)))

        self.logger.info("model architecture: {}, learning rate: {}".format("AlexNet", self.learning_rate))

        filename = self.logger.root.handlers[0].baseFilename
        self.logger.info("model details: ")
        with open(filename, "a") as f:
            with redirect_stdout(f):
                self.model.summary()

    def train_local_model(self, batch_size=32, local_epochs=15):
        """
        Train one client with its own data for one fed-epoch.
        """
        # The data held by each participant should be divided into tow parts:
        # train set and test set, both of which are used to train the local model.
        assert self.current_cid != -1, "Forget to register the current cid during federated training!"
        train_dataset = self.dataset.train[self.current_cid]
        valid_dataset = self.dataset.test

        if len(train_dataset.x) <= len(valid_dataset.x):
            size = len(train_dataset.x)
        else:
            size = len(valid_dataset.x)

        train_features, train_labels = train_dataset.x[: size], train_dataset.y[: size]
        valid_features, valid_labels = valid_dataset.x[: size], valid_dataset.y[: size]


        train_data_batches = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(batch_size)
        valid_data_batches = tf.data.Dataset.from_tensor_slices((valid_features, valid_labels)).batch(batch_size)

        for epoch in range(local_epochs):
            self.model.reset_metrics()

            # learning rate scheduler
            if epoch == 25:
                self.model.optimizer.lr.assign(self.model.optimizer.lr / 10)
            elif epoch == 60:
                self.model.optimizer.lr.assign(self.model.optimizer.lr / 10)

            train_result = None
            for (features, labels) in train_data_batches:
                train_result = self.model.train_on_batch(features, labels)

            valid_result = None
            for (features, labels) in valid_data_batches:
                valid_result = self.model.test_on_batch(features, labels)

            print("local epoch: {}/{}, "
                  "learning rate: {}".format((epoch + 1), local_epochs, self.model.optimizer.lr.numpy()))
            print("train set: {}, valid set: {}".format(len(train_labels), len(valid_labels)))
            print("train: {}".format(dict(zip(self.model.metrics_names, train_result))))
            print("valid: {}".format(dict(zip(self.model.metrics_names, valid_result))))

            self.logger.info("local epoch: {}/{}, "
                             "learning rate: {}".format((epoch + 1), local_epochs, self.model.optimizer.lr.numpy()))
            self.logger.info("train set: {}, valid set: {}".format(len(train_labels), len(valid_labels)))
            self.logger.info("train: {}".format(dict(zip(self.model.metrics_names, train_result))))
            self.logger.info("valid: {}".format(dict(zip(self.model.metrics_names, valid_result))))

    def upload_local_parameters(self):
        """ Return all of the variables list"""
        assert self.current_cid != -1 or self.isolated_cid != -1, "Forget to register the current cid and isolated cid!"

        if self.current_cid == self.isolated_cid:
            size = len(self.model.variables)
            self.isolated_local_parameters = [[]] * size
            for index in range(size):
                self.isolated_local_parameters[index] = self.model.variables[index].numpy()

        return self.model.trainable_variables

    def download_global_parameters(self, global_vars):
        """ Assign all of the variables with global vars """
        # The federated learning environment is just established.
        assert self.current_cid != -1 or self.isolated_cid != -1, "Forget to register the current cid and isolated cid!"

        if global_vars is None:
            # Clear the parameters.
            self.compile_model()
            return

        client_vars = self.model.trainable_variables

        if self.current_cid == self.isolated_cid:
            assert self.isolated_local_parameters, "Isolated local are not initialized!"
            for var, value in zip(client_vars, self.isolated_local_parameters):
                var.assign(value)

        for var, value in zip(client_vars, global_vars):
            var.assign(value)

    def choose_clients(self, ratio=1.0):
        """ Randomly choose some clients """
        client_num = self.get_clients_num()
        # choose_num = math.floor(client_num * ratio)
        choose_num = math.ceil(client_num * ratio)
        # return np.random.permutation(client_num)[:choose_num]

        return list(range(choose_num))

    def get_clients_num(self):

        return len(self.dataset.train)
