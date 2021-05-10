import copy
import yaml
from collections import namedtuple

from membership_inference_attack.utils.logger import create_attacker_logger
from membership_inference_attack.utils.data_handler import AttackerDataHandler, VerifierDataHandler
from membership_inference_attack.membership_inference_attack import MembershipInferenceAttack


ATTACK_MSG = namedtuple("ATTACK_MSG", "attack_type, target_cid, target_fed_epoch")

with open("./demo/hyper_parameters.yaml", mode='r', encoding="utf-8") as f:
    hyper_parameters = yaml.load(f, Loader=yaml.FullLoader)


class Attacker:
    def __init__(self, cid, local_epochs):
        self.cid = cid
        self.local_epochs = local_epochs
        self.logger = create_attacker_logger()
        self.log_info()

        self.attack_msg = None
        self.attacker_data_handler = None
        self.verifier_data_handler = None
        self.membership_inference = None
        self.inference_model = None

    def log_info(self):
        self.logger.info("attacker cid: {}, attacker local epochs: {}".format(self.cid, self.local_epochs))

    def declare_attack(self, attack_type, target_cid, target_fed_epoch):
        self.attack_msg = ATTACK_MSG(attack_type, target_cid, target_fed_epoch)

    def generate_attacker_data_handler(self, client):
        target_cid = self.attack_msg.target_cid
        train_data = client.dataset.train[self.cid]
        test_data = client.dataset.test
        visual_member_data = client.dataset.train[target_cid]

        attacker_data_handler_config = hyper_parameters["attacker_data_handler"]
        batch_size = attacker_data_handler_config["batch_size"]
        train_ratio = attacker_data_handler_config["train_ratio"]
        exposed_percentage = attacker_data_handler_config["exposed_percentage"]

        self.attacker_data_handler = AttackerDataHandler(test_data=copy.deepcopy(test_data),
                                                         train_data=copy.deepcopy(train_data),
                                                         exposed_percentage=exposed_percentage,
                                                         train_ratio=train_ratio,
                                                         batch_size=batch_size,
                                                         input_shape=client.input_shape,
                                                         logger=self.logger)

        self.verifier_data_handler = VerifierDataHandler(member_target_data=copy.deepcopy(visual_member_data),
                                                         nonmember_target_data=copy.deepcopy(test_data),
                                                         batch_size=batch_size)

    def create_membership_inference_model(self, client):
        target_model = client.model

        inference_model_config = hyper_parameters["inference_model"]
        epochs = inference_model_config["epochs"]
        learning_rate = inference_model_config["learning_rate"]
        optimizer_name = inference_model_config["optimizer_name"]
        exploit_label = inference_model_config["exploit_label"]
        exploit_loss = inference_model_config["exploit_loss"]
        exploited_layer_indexes = inference_model_config["exploited_layer_indexes"]
        exploited_gradient_indexes = inference_model_config["exploited_gradient_indexes"]

        self.membership_inference = MembershipInferenceAttack(local_model=target_model,
                                                              attacker_data_handler=self.attacker_data_handler,
                                                              exploited_layer_indexes=exploited_layer_indexes,
                                                              exploited_gradient_indexes=exploited_gradient_indexes,
                                                              exploit_label=exploit_label,
                                                              exploit_loss=exploit_loss,
                                                              learning_rate=learning_rate,
                                                              epochs=epochs,
                                                              optimizer_name=optimizer_name,
                                                              attack_msg=self.attack_msg,
                                                              logger=self.logger)
        self.inference_model = self.membership_inference.inference_model

    def train_inference_model(self):
        self.membership_inference.train_inference_model()

    def test_inference_model(self, client):
        self.membership_inference.visually_test_inference_model(target_model=client.model,
                                                                verifier_data_handler=self.verifier_data_handler)
