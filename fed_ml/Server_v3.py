"""Server"""
import copy
import numpy as np

class Server:
    def __init__(self):
        self.global_parameters = None
        self.local_parameters_sum = None

    def initialize_local_parameters_sum(self):
        self.local_parameters_sum = None

    def accumulate_local_parameters(self, current_local_parameters):
        if self.local_parameters_sum is None:
            self.local_parameters_sum = copy.deepcopy(current_local_parameters)
            size = len(self.local_parameters_sum)
            for i in range(size):
                self.local_parameters_sum[i] = current_local_parameters[i].numpy()
            return
        # Mutable objects are passed by reference.
        for paras_sum, paras in zip(self.local_parameters_sum, current_local_parameters):
            paras_sum += paras.numpy()

    def update_global_parameters(self, client_num):
        if self.global_parameters is None:
            self.global_parameters = self.local_parameters_sum
        size = len(self.local_parameters_sum)
        for i in range(size):
            self.global_parameters[i] = self.local_parameters_sum[i] / client_num
        # for global_paras, local_paras_sum in zip(self.global_parameters, self.local_parameters_sum):
        #     global_paras = local_paras_sum / client_num