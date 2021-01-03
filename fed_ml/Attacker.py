import copy
from collections import namedtuple
import tensorflow as tf

import ml_privacy_meter
from ml_privacy_meter.utils.losses import CrossEntropyLoss

ATTACK_MSG = namedtuple("ATTACK_MSG", "attack_type, cid, fed_ep")

class Attacker:
    def __init__(self):
        self.attack_msg = None
        self.data_handler = None
        self.target_member_features = None
        self.target_member_labels = None
        gpus = tf.config.experimental.list_physical_devices('GPU')
        self.gpu = gpus[0].name

    def declare_attack(self, attack_type, cid, fed_ep):
        self.attack_msg = ATTACK_MSG(attack_type, cid, fed_ep)

    def generate_attack_data(self, client, attack_percentage=10):
        train_data = client.dataset.train[self.attack_msg.cid]
        test_data = client.dataset.test
        self.data_handler = ml_privacy_meter.utils.attack_data_v2.AttackData(test_data=copy.deepcopy(test_data),
                                                                             train_data=copy.deepcopy(train_data),
                                                                             batch_size=32,
                                                                             attack_percentage=attack_percentage,
                                                                             input_shape=client.input_shape)

    def generate_target_gradient(self, client, instances_num=1):    # 100， 10， 50， 5, 1, 3
        self.target_member_features = self.data_handler.exposed_features[: instances_num]
        self.target_member_labels = self.data_handler.exposed_labels[: instances_num]
        with tf.GradientTape() as tape:
            logits = client.model(self.target_member_features)
            loss = CrossEntropyLoss(logits, self.target_member_labels)
            # loss = tf.reduce_mean(loss)
        target_var = client.model.trainable_variables
        self.target_gradients = copy.deepcopy(tape.gradient(loss, target_var))

    def craft_global_parameters(self, parameters, update_rate=0.5): # 1.0, 0.1, 0.5, 0.001
        size = len(parameters)
        for i in range(size):
            parameters[i] += update_rate * self.target_gradients[i].numpy()

    def craft_adversarial_parameters(self, client, update_rate=0.5):
        for var, value in zip(client.model.trainable_variables, self.target_gradients):
            var.assign_add(update_rate * value)

    def membership_inference_attack(self, client):
        target_model = client.model
        attackobj = ml_privacy_meter.attack.meminf.initialize(
            target_train_model=target_model,
            target_attack_model=target_model,
            train_datahandler=self.data_handler,
            attack_datahandler=self.data_handler,
            layers_to_exploit=[6],
            gradients_to_exploit=[6],
            device=self.gpu, epochs=10,
            attack_msg=self.attack_msg,
            model_name=self.attack_msg.attack_type)
        attackobj.train_attack()
        # attackobj.test_attack()

