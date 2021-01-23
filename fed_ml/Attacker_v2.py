import copy
import yaml
from collections import namedtuple

from membership_inference_attack.utils.attacker_data_handler import *
from membership_inference_attack.membership_inference_attack import *

ATTACK_MSG = namedtuple("ATTACK_MSG", "attack_type, target_cid, target_fed_epoch")

with open("demo\hyper_parameters.yaml", mode='r', encoding="utf-8") as f:
    hyper_parameters = yaml.load(f, Loader=yaml.FullLoader)


class Attacker:
    def __init__(self):
        self.attack_msg = None
        self.data_handler = None
        self.membership_inference = None
        self.inference_model = None

    def declare_attack(self, attack_type, target_cid, target_fed_epoch):
        self.attack_msg = ATTACK_MSG(attack_type, target_cid, target_fed_epoch)

    def generate_attacker_data_handler(self, client):
        test_data = client.dataset.test
        train_data = client.dataset.train[self.attack_msg.target_cid]

        attacker_data_handler_info = hyper_parameters["attacker_data_handler"]
        batch_size = attacker_data_handler_info["batch_size"]
        exposed_percentage = attacker_data_handler_info["exposed_percentage"]

        self.data_handler = AttackerDataHandler(test_data=copy.deepcopy(test_data),
                                                train_data=copy.deepcopy(train_data),
                                                batch_size=batch_size,
                                                exposed_percentage=exposed_percentage,
                                                input_shape=client.input_shape)

    def generate_membership_inference_model(self, client):
        target_model = client.model

        inference_model_info = hyper_parameters["inference_model"]
        epochs = inference_model_info["epochs"]
        learning_rate = inference_model_info["learning_rate"]
        optimizer_name = inference_model_info["optimizer_name"]
        exploit_label = inference_model_info["exploit_label"]
        exploit_loss = inference_model_info["exploit_loss"]
        exploited_layer_indexes = inference_model_info["exploited_layer_indexes"]
        exploited_gradient_indexes = inference_model_info["exploited_gradient_indexes"]

        self.membership_inference = MembershipInferenceAttack(target_model=target_model,
                                                              attacker_data_handler=self.data_handler,
                                                              exploited_layer_indexes=exploited_layer_indexes,
                                                              exploited_gradient_indexes=exploited_gradient_indexes,
                                                              exploit_label=exploit_label,
                                                              exploit_loss=exploit_loss,
                                                              learning_rate=learning_rate,
                                                              epochs=epochs,
                                                              optimizer_name=optimizer_name)
        self.inference_model = self.membership_inference.inference_model

    def train_inference_model(self):
        self.membership_inference.train_inference_model()

    def test_inference_model(self):
        self.membership_inference.test_inference_model()

