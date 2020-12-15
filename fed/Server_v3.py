"""Server"""

class Server:
    def __init__(self):
        self.global_parameters = None
        self.local_parameters_sum = None

    def initialize_local_parameters_sum(self):
        self.local_parameters_sum = None

    def accumulate_local_parameters(self, current_local_parameters):
        if self.local_parameters_sum is None:
            self.local_parameters_sum = current_local_parameters
            return
        # Mutable objects are passed by reference.
        for paras_sum, paras in zip(self.local_parameters_sum, current_local_parameters):
            paras_sum.assign_add(paras)

    def update_global_parameters(self, client_num):
        if self.global_parameters is None:
            self.global_parameters = self.local_parameters_sum
        size = len(self.local_parameters_sum)
        for i in range(size):
            self.global_parameters[i] = self.local_parameters_sum[i] / client_num