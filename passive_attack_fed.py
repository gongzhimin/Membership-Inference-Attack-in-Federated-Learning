import copy

from fed_ml.Client_v3 import Clients
from fed_ml.Server_v3 import Server
from fed_ml.Model_v2 import alexnet
import ml_privacy_meter


def passive_attack(client, client_id):
    train_data = client.dataset.train[client_id]
    test_data = client.dataset.test
    # test_data and train_data are mutable objects,
    # hence, they will be changed after passing to `attack_data` method directly.
    data_handler = ml_privacy_meter.utils.attack_data_v2.attack_data(test_data=copy.deepcopy(test_data),
                                                                     train_data=copy.deepcopy(train_data),
                                                                     batch_size=32,
                                                                     attack_percentage=10,
                                                                     input_shape=client.input_shape)
    # The application of shadow models has been confirmed:
    # https://github.com/privacytrustlab/ml_privacy_meter/issues/19
    # The answer of coder confused me...
    target_model = client.model
    shadow_model = alexnet(input_shape, client.classes_num)
    attackobj = ml_privacy_meter.attack.meminf.initialize(
        # target_train_model=shadow_model,
        target_train_model=target_model,
        target_attack_model=target_model,
        train_datahandler=data_handler,
        attack_datahandler=data_handler,
        layers_to_exploit=[6],
        # gradients_to_exploit=[6],
        device=None, epochs=10, model_name="without gradients")
    attackobj.train_attack()
    attackobj.test_attack()


if __name__ == "__main__":
    """Set hyper-parameters."""
    epoch = 5
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

    """Begin training."""
    for ep in range(epoch):
        # Empty local_parameters_sum at the beginning of each epoch.
        server.initialize_local_parameters_sum()
        # Choose a random selection of active clients to train in this epoch.
        active_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
        # Train these clients.
        for client_id in active_clients:
            print("[fed-epoch {}] cid: {}".format((ep + 1), client_id))
            # In each epoch, clients download parameters from the server,
            # and then train local models to adapt their parameters.
            client.download_global_parameters(server.global_parameters)
            # Perform passive local membership inference attack, since only get global parameters.
            # if client_id == 1:
            #     print("passive local attack on cid: {} in fed_ml-epoch: {}".format((ep+1), client_id))
            #     passive_attack(client, client_id)
            client.train_epoch(cid=client_id)
            # Accumulate local parameters.
            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)
            # Perform passive global membership inference attack, since the target model's parameters are informed.
            if client_id == 1:
                print("passive global attack on cid: {} in fed-epoch: {}".format((ep + 1), client_id))
                passive_attack(client, client_id)
        # Update global parameters in each epoch.
        server.update_global_parameters(len(active_clients))
