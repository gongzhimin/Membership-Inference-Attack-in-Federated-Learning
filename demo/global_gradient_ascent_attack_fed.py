import yaml

from fed_ml.Client import Clients
from fed_ml.Server import Server
from fed_ml.Attacker import Attacker

with open("hyper_parameters.yaml", mode='r', encoding="utf-8") as f:
    hyper_parameters = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    """Set hyper-parameters."""
    epoch = hyper_parameters["participants"]["fed_epochs"]  # 50, 100
    learning_rate = hyper_parameters["participants"]["learning_rate"]
    # The ml_privacy_meter can't handle the scenario with too many participants.
    CLIENT_NUMBER = hyper_parameters["participants"]["clients_num"]
    # And as federated learning is online,
    # participants are uncertain about their online status in each training epoch.
    CLIENT_RATIO_PER_ROUND = hyper_parameters["participants"]["client_ratio_per_round"]
    # Some characteristics of the dataset cifar-100.
    input_shape = tuple(hyper_parameters["dataset"]["input_shape"])
    # classes_num = 100   # cifar-100
    classes_num = hyper_parameters["dataset"]["classes_num"]  # cifar-10

    """Build clients, server and attacker."""
    client = Clients(input_shape=input_shape,
                     classes_num=classes_num,
                     learning_rate=learning_rate,
                     clients_num=CLIENT_NUMBER,
                     dataset_path="./datasets/cifar100.txt")
    server = Server()
    attacker = Attacker()

    """Target the attack."""
    target_cid = hyper_parameters["participants"]["target_cid"]
    target_ep = hyper_parameters["participants"]["target_fed_epoch"]  # 45, 95
    attacker_cid = hyper_parameters["attacker_participant"]["attacker_cid"]
    attacker.declare_attack("GGAA", target_cid, target_ep)
    attacker.generate_attack_data(client)

    """Begin training."""
    for ep in range(epoch):
        # Empty local_parameters_sum at the beginning of each epoch.
        server.initialize_local_parameters_sum()
        # Choose a random selection of active clients to train in this epoch.
        active_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
        # Train these clients.
        for client_id in active_clients:
            client.current_cid = client_id
            print("[fed-epoch {}] cid: {}".format(ep, client_id))
            # The attacker repeats gradient ascend algorithm for each epoch of the training.
            # if ep == target_ep and client_id == target_cid:
            #     print("Craft the global parameters received by cid: {} in fed-epoch: {}".format(client_id, ep))
            #     client.download_global_parameters(server.global_parameters)
            #     attacker.generate_target_gradient(client)
            #     attacker.craft_global_parameters(server.global_parameters)
            client.download_global_parameters(server.global_parameters)
            if client_id != attacker_cid:
                client.train_local_model(local_epochs=hyper_parameters["participants"]["local_epochs"])
            else:
                client.train_local_model(local_epochs=hyper_parameters["attacker_participant"]["local_epochs"])
            # Accumulate local parameters.
            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)
            if ep == target_ep and client_id == target_cid:
                print("global gradient ascent attack on cid: {} in fed-epoch: {}".format(client_id, ep))
                # attacker.declare_attack("GGAA", target_cid, ep)
                attacker.membership_inference_attack(client, is_gradient_ascent=False)
        # Update global parameters in each epoch.
        server.update_global_parameters(len(active_clients))
