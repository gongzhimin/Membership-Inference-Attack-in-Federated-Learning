from Dataset_v2 import Dataset
import math
from collections import namedtuple
import numpy as np
import tensorflow as tf

"""
In the third version of `Client.py`, the tensorflow relied on updates to version 2, 
which really different from version 1 with the graph execution. 
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


        self.dataset = Dataset(dataset_path, split=clients_num)
        # initialize the status of activate attack
        self.is_crafted = False
        self.craft_id = 0
        self.hashed_crafted_records = []
        self.labels_crafted = []

    def run_test(self, num):
        pass

    def train_epoch(self, cid, batch_size=32, dropout_rate=0.5):
        """
        Train one client with its own data for one epoch.
        And we leave a back door at here.
        cid: Client id
        """
        dataset = self.dataset.train[cid]
        pass

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
        pass

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        pass


    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        client_num = self.get_clients_num()
        choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        return len(self.dataset.train)