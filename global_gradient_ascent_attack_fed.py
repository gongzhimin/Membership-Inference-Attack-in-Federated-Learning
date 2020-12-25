from fed_ml.Client import Clients
from fed_ml.Server import Server
from fed_ml.Attacker import Attacker


if __name__ == "__main__":
    """Set hyper-parameters."""
    epoch = 6
    learning_rate = 0.0001
    # The ml_privacy_meter can't handle the scenario with too many participants.
    CLIENT_NUMBER = 5
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

    """Begin training."""
    for ep in range(epoch):
        # Empty local_parameters_sum at the beginning of each epoch.
        server.initialize_local_parameters_sum()
        # Choose a random selection of active clients to train in this epoch.
        active_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
        # Train these clients.
        for client_id in active_clients:
            client.current_cid = client_id
            print("[fed-epoch {}] cid: {}".format((ep + 1), client_id))
            if client_id == target_cid and ep == 2:
                attacker.craft_global_parameters(server.global_parameters)
                print("The global parameters have been crafted.")
            client.download_global_parameters(server.global_parameters)
            client.train_epoch()
            # Accumulate local parameters.
            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)
            if client_id == target_cid and ep == 1:
                attacker.declare_attack("GGAA", client_id, (ep + 2))
                attacker.generate_attack_data(client)
                attacker.generate_target_gradient(client)
            if client_id == target_cid and ep == 2:
                print("global gradient ascent attack on cid: {} in fed-epoch: {}".format(client_id, (ep + 1)))
                attacker.membership_inference_attack(client)
        # Update global parameters in each epoch.
        server.update_global_parameters(len(active_clients))
