import os
import sys
import yaml
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fed_exchange_weight_bias.client import Clients
from fed_exchange_weight_bias.server import Server
from membership_inference_attack.attacker import Attacker
from fed_exchange_weight_bias.utils.logger import initialize_logging, create_federated_logger
from demo.utils import capture_cmdline, map_mia


with open("./demo/hyper_parameters.yaml", mode='r', encoding="utf-8") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":
    params = capture_cmdline(params)

    dataset = params["dataset"]
    model_name = params["model"]
    attack_name = params["attack_name"]
    dataset_config = params[dataset]
    input_shape = dataset_config["input_shape"]
    classes_num = dataset_config["classes_num"]
    data_dir = dataset_config["data_dir"]

    participant_config = params["participant"]
    fed_epochs = participant_config["fed_epochs"]
    learning_rate = participant_config["learning_rate"]
    clients_num = participant_config["clients_num"]
    client_ratio_per_round = participant_config["client_ratio_per_round"]
    train_ratio = participant_config["train_ratio"]

    target_participant_config = params["target_participant"]
    target_cid = target_participant_config["target_cid"]
    target_fed_epoch = target_participant_config["target_fed_epoch"]

    isolated_participant_config = params["isolated_participant"]
    isolated_cid = isolated_participant_config["isolated_cid"]

    attacker_participant_config = params["attacker_participant"]
    attacker_cid = attacker_participant_config["attacker_cid"]
    attacker_local_epochs = attacker_participant_config["local_epochs"]

    client = Clients(dataset=dataset,
                     data_dir=data_dir,
                     model_name=model_name,
                     input_shape=input_shape,
                     classes_num=classes_num,
                     learning_rate=learning_rate,
                     clients_num=clients_num,
                     train_ratio=train_ratio)
    server = Server()
    attacker = Attacker(cid=attacker_cid,
                        local_epochs=attacker_local_epochs)

    attack_type = "{}_{}_{}".format(attack_name, model_name, dataset)
    initialize_logging(filepath="logs/{}/".format(attack_type),
                       filename="{}.log".format(attack_type))
    federated_logger = create_federated_logger(attack_name)

    client.isolated_cid = isolated_cid
    federated_logger.info("isolated participant (cid): {}".format(isolated_cid))

    attacker.declare_attack(attack_type, target_cid, target_fed_epoch)
    attacker.generate_attacker_data_handler(client)

    for epoch in range(fed_epochs):
        server.initialize_local_parameters_sum()
        activated_cid_list = client.choose_clients(client_ratio_per_round)

        for cid in activated_cid_list:
            client.current_cid = cid

            map_mia(attack_name, epoch, cid, server,
                    client, attacker, params, federated_logger)

        server.update_global_parameters(len(activated_cid_list))
