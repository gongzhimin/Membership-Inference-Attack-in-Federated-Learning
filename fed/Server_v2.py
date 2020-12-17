import tensorflow as tf
import copy
# from tqdm import tqdm

# from Client_v3 import Clients
# from Model_v2 import classification_cnn
# import sys
# sys.path.append("../")
import ml_privacy_meter
from fed.Client_v3 import Clients
from fed.Model_v2 import alexnet


def buildClients(num):
    learning_rate = 0.0001
    num_input = 32
    num_input_channel = 3
    num_classes = 100

    #create Client and model
    return Clients(input_shape=(32, 32, 3),
                  classes_num=num_classes,
                  learning_rate=learning_rate,
                  clients_num=num)


def run_global_test(client, global_vars, test_num):
    client.download_global_parameters(global_vars)
    acc, loss = client.run_test(test_num)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))

def accumulate_local_parameters(client, client_vars_sum):
    current_client_vars = client.upload_local_parameters()
    if client_vars_sum is None:
        return current_client_vars
    for cv, ccv in zip(client_vars_sum, current_client_vars):
        cv.assign_add(ccv)
    return client_vars_sum

def update_global_parameters(global_vars, client_vars_sum, client_num):
    if global_vars == None:
        global_vars = client_vars_sum
    size = len(client_vars_sum)
    for i in range(size):
        global_vars[i] = client_vars_sum[i] / client_num
    return global_vars



#### SOME TRAINING PARAMS ####
CLIENT_NUMBER = 5   # The ml_privacy_meter can't handle the scenario with too many participants.
CLIENT_RATIO_PER_ROUND = 1.00
# epoch = 360
epoch = 60   # during debugging
input_shape = (32, 32, 3)   # the type of image in cifar-100

#### CREATE CLIENT AND LOAD DATASET ####
client = buildClients(CLIENT_NUMBER)

#### BEGIN TRAINING ####
# global_vars = client.update_local_parameters()
global_vars = None
for ep in range(epoch):
    # We are going to sum up active clients' vars at each epoch
    client_vars_sum = None

    # Choose some clients that will train on this epoch
    random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)

    # Train with these clients
    for client_id in random_clients:
        print("[fed-epoch {}] cid: {}".format((ep + 1), client_id))
        # In each epoch, clients download parameters from the server
        # and then train the local model to update their parameters
        client.download_global_parameters(global_vars)
        # Perform the craft
        # if ep == 2 and client_id == 2:
        #     client.craft(cid=client_id)
        # else:
        #     client.train_epoch(cid=client_id)
        client.train_epoch(cid=client_id)

        # A passive inference attack can be performed here after crafting
        if ep == 1 and client_id == 0:
            print("[fed-epoch {}] attack cid: {}".format((ep + 1), client_id))
            train_data = client.dataset.train[client_id]
            test_data = client.dataset.test
            # test_data and train_data are mutable objects,
            # hence, they will be changed after passing to `attack_data` method directly.
            data_handler = ml_privacy_meter.utils.attack_data_v2.attack_data(test_data=copy.deepcopy(test_data),
                                                                          train_data=copy.deepcopy(train_data),
                                                                          batch_size=32,
                                                                          attack_percentage=10, input_shape=input_shape)
            # The use of shadow models has been confirmed:
            # https://github.com/privacytrustlab/ml_privacy_meter/issues/19
            target_model = client.model
            target_model.summary()
            shadow_model = alexnet(input_shape)
            attackobj = ml_privacy_meter.attack.meminf_v2.initialize(
                target_train_model=shadow_model,
                target_attack_model=target_model,
                train_datahandler=data_handler,
                attack_datahandler=data_handler,
                layers_to_exploit=[6],
                # gradients_to_exploit=[6],
                device=None, epochs=10, model_name='blackbox1')
            attackobj.train_attack()
            attackobj.test_attack()

        # Accumulate local parameters.
        client_vars_sum = accumulate_local_parameters(client, client_vars_sum)

    # Update global parameters
    global_vars = update_global_parameters(global_vars, client_vars_sum, len(random_clients))


    # run test on 1000 instances
    # run_global_test(client, global_vars, test_num=600)


#### FINAL TEST ####
# run_global_test(client, global_vars, test_num=1000)
