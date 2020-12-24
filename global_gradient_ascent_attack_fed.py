import copy
from collections import namedtuple

from fed_ml.Client import Clients
from fed_ml.Server import Server
import ml_privacy_meter


ATTACK_MSG = namedtuple("ATTACK_MSG", "attack_type, cid, fed_ep")

def passive_attack(client, attack_msg):
    train_data = client.dataset.train[attack_msg.cid]
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
        # gradients_to_exploit=[6],
        device=None, epochs=10,
        attack_msg = attack_msg, model_name="meminf_fed")
    attackobj.train_attack()
    attackobj.test_attack()

def craft_global_parameters(global_parameters):
    pass


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

    """Build clients and server."""
    client = Clients(input_shape=input_shape,
                    classes_num=classes_num,
                    learning_rate=learning_rate,
                    clients_num=CLIENT_NUMBER,
                     dataset_path="./datasets/cifar100.txt")
    server = Server()

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
                craft_global_parameters(server.global_parameters)
                print("The global parameters have been crafted.")
            client.download_global_parameters(server.global_parameters)
            client.train_epoch()
            # Accumulate local parameters.
            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)
            # Perform passive global membership inference attack, since the target model's parameters are informed.
            if client_id == target_cid and ep == 2:
                attack_msg = ATTACK_MSG(attack_type="GGAA", cid=client_id, fed_ep=ep+1)
                print("global gradient ascent attack on cid: {} in fed-epoch: {}".format(client_id, (ep + 1)))
                passive_attack(client, attack_msg)
        # Update global parameters in each epoch.
        server.update_global_parameters(len(active_clients))
