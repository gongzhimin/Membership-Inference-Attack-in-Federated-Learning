from fed_ml.Client import Clients
from fed_ml.Server import Server
from fed_ml.Attacker import Attacker


if __name__ == "__main__":
    """Set hyper-parameters."""
    epoch = 100  # 50, 100
    learning_rate = 0.0001
    # The ml_privacy_meter can't handle the scenario with too many participants.
    CLIENT_NUMBER = 4
    # And as federated learning is online,
    # participants are uncertain about their online status in each training epoch.
    CLIENT_RATIO_PER_ROUND = 1.00
    # Some characteristics of the dataset cifar-100.
    input_shape = (32, 32, 3)
    # classes_num = 100   # cifar-100
    classes_num = 10    # cifar-10

    """Build clients, server and attacker."""
    client = Clients(input_shape=input_shape,
                    classes_num=classes_num,
                    learning_rate=learning_rate,
                    clients_num=CLIENT_NUMBER,
                     dataset_path="./datasets/cifar100.txt")
    server = Server()
    attacker = Attacker()

    """Target the attack."""
    target_cid = 1
    target_ep = 95  # 45, 95
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
            client.train_local_model(local_epochs=1)
            # Accumulate local parameters.
            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)
            if ep == target_ep  and client_id == target_cid:
                print("global gradient ascent attack on cid: {} in fed-epoch: {}".format(client_id, ep))
                # attacker.declare_attack("GGAA", target_cid, ep)
                attacker.membership_inference_attack(client, is_gradient_ascent=True)
        # Update global parameters in each epoch.
        server.update_global_parameters(len(active_clients))
