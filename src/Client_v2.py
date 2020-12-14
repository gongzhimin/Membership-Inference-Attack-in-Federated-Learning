from Dataset_v2 import Dataset
from Model import AlexNet
import math
from collections import namedtuple
import numpy as np
import tensorflow as tf


# The definition of fed model with a named tuple. What an amazing idea!
FedModel = namedtuple("FedModel", "X Y DROP_RATE train_op loss_op acc_op loss prediction grads")


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
        self.graph = tf.compat.v1.Graph()
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

        # Call the create function to build the computational graph of AlexNet
        net = AlexNet(input_shape, num_classes, learning_rate, self.graph)
        # self.model_object = net # one of the inputs to inference attack
        self.model = FedModel(*net)

        # initialize
        with self.graph.as_default():
            self.init_op = tf.compat.v1.global_variables_initializer()
            self.sess.run(self.init_op)

        self.dataset = Dataset(dataset_path, split=clients_num)
        # initialize the status of activate attack
        self.is_crafted = False
        self.craft_id = 0
        self.hashed_crafted_records = []
        self.labels_crafted = []

    def run_test(self, num):
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test.next_batch(num)
            feed_dict = {
                self.model.X: batch_x,
                self.model.Y: batch_y,
                self.model.DROP_RATE: 0
            }
        return self.sess.run([self.model.acc_op, self.model.loss_op],
                             feed_dict=feed_dict)

    def train_epoch(self, cid, batch_size=32, dropout_rate=0.5):
        """
        Train one client with its own data for one epoch.
        And we leave a back door at here.
        cid: Client id
        """
        dataset = self.dataset.train[cid]
        with self.graph.as_default():
            for _ in range(math.ceil(dataset.size / batch_size)):
                batch_x, batch_y = dataset.next_batch(batch_size)
                feed_dict = {
                    self.model.X: batch_x,
                    self.model.Y: batch_y,
                    self.model.DROP_RATE: dropout_rate
                }
                self.sess.run(self.model.train_op, feed_dict=feed_dict)

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
        with self.graph.as_default():
            feed_dict = {
                self.model.X: total_x,
                self.model.Y: total_y,
                self.model.DROP_RATE: dropout_rate
            }
            self.sess.run(self.model.train_op, feed_dict=feed_dict)

    def get_client_vars(self):
        """ Return all of the variables list"""
        with self.graph.as_default():
            client_vars = self.sess.run(tf.compat.v1.trainable_variables())
        return client_vars

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        with self.graph.as_default():
            all_vars = tf.compat.v1.trainable_variables()
            for variable, value in zip(all_vars, global_vars):
                variable.load(value, self.sess)


    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        client_num = self.get_clients_num()
        choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        return len(self.dataset.train)
