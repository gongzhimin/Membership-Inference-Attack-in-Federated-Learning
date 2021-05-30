import copy
from collections import namedtuple

import ml_privacy_meter

ATTACK_MSG = namedtuple("ATTACK_MSG", "attack_type, cid, fed_ep")

class Attacker:
    def __init__(self, attack_type, cid, fed_ep):
        self.attack_msg = ATTACK_MSG(attack_type, cid, fed_ep)

    def membership_inference_attack(self, client):
        train_data = client.dataset.train[self.attack_msg.cid]
        test_data = client.dataset.test
        data_handler = ml_privacy_meter.utils.attack_data_v2.attack_data(test_data=copy.deepcopy(test_data),
                                                                         train_data=copy.deepcopy(train_data),
                                                                         batch_size=32,
                                                                         attack_percentage=10,
                                                                         input_shape=client.input_shape)
        target_model = client.model
        attackobj = ml_privacy_meter.attack.meminf.initialize(
            target_train_model=target_model,
            target_attack_model=target_model,
            train_datahandler=data_handler,
            attack_datahandler=data_handler,
            layers_to_exploit=[6],
            gradients_to_exploit=[6],
            device=None, epochs=10,
            attack_msg=self.attack_msg, model_name="meminf_fed")
        attackobj.train_attack()
        attackobj.test_attack()

