from fed_exchange_weight_bias.client import *
from fed_exchange_weight_bias.server import *
from fed_exchange_weight_bias.utils.logger import *
from membership_inference_attack.attacker import *

with open("hyper_parameters.yaml", mode='r', encoding="utf-8") as f:
    hyper_parameters = yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":
    dataset = hyper_parameters["dataset"]
    model_name = hyper_parameters["model"]
    attack_type = "local_passive_attack_{}_{}".format(model_name, dataset)
    dataset_config = hyper_parameters[dataset]
    input_shape = dataset_config["input_shape"]
    classes_num = dataset_config["classes_num"]
    data_dir = dataset_config["data_dir"]

    participant_config = hyper_parameters["participant"]
    fed_epochs = participant_config["fed_epochs"]
    learning_rate = participant_config["learning_rate"]
    batch_size = participant_config["batch_size"]
    clients_num = participant_config["clients_num"]
    client_ratio_per_round = participant_config["client_ratio_per_round"]
    client_local_epochs = participant_config["local_epochs"]
    train_ratio = participant_config["train_ratio"]

    target_participant_config = hyper_parameters["target_participant"]
    target_cid = target_participant_config["target_cid"]
    target_fed_epoch = target_participant_config["target_fed_epoch"]

    isolated_participant_config = hyper_parameters["isolated_participant"]
    isolated_cid = isolated_participant_config["isolated_cid"]

    attacker_participant_config = hyper_parameters["attacker_participant"]
    attacker_cid = attacker_participant_config["attacker_cid"]
    attacker_local_epochs = attacker_participant_config["local_epochs"]

    initialize_logging(filepath="logs/{}/".format(attack_type), filename="local_passive_attack.log")
    federated_logger = create_federated_logger("local passive attack")


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

    client.isolated_cid = isolated_cid
    federated_logger.info("isolated participant (cid): {}".format(isolated_cid))

    attacker.declare_attack(attack_type, target_cid, target_fed_epoch)
    attacker.generate_attacker_data_handler(client)

    for epoch in range(fed_epochs):
        server.initialize_local_parameters_sum()
        activated_cid_list = client.choose_clients(client_ratio_per_round)

        for cid in activated_cid_list:
            client.current_cid = cid

            if epoch == target_fed_epoch and cid == target_cid:
                print("train inference model on victim (cid): {} "
                      "at federated learning epoch: {}".format(target_cid, (target_fed_epoch + 1)))
                federated_logger.info("train inference model on victim (cid): {}, "
                                      "federated training epoch: {}".format(target_cid, (target_fed_epoch + 1)))
                attacker.create_membership_inference_model(client)
                attacker.train_inference_model()
                attacker.test_inference_model(client)

            print("[federated learning epoch: {}, current participant (cid): {}]".format((epoch + 1), cid))
            federated_logger.info("federated training epoch: {}, "
                                  "current participant (cid): {}".format((epoch + 1), cid))

            client.download_global_parameters(server.global_parameters)

            if cid == attacker_cid:
                client.train_local_model(batch_size=batch_size, local_epochs=attacker_local_epochs)
            else:
                client.train_local_model(batch_size=batch_size, local_epochs=client_local_epochs)

            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)

        server.update_global_parameters(len(activated_cid_list))
