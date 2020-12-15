import tensorflow as tf
# import sys
# sys.path.append("../membership_inference_attack/")
# import ml_privacy_meter
from tqdm import tqdm

from Client_v3 import Clients

# tf.compat.v1.disable_eager_execution()
def buildClients(num):
    learning_rate = 0.0001
    num_input = 32
    num_input_channel = 3
    num_classes = 100

    #create Client and model
    return Clients(input_shape=(32, 32, 3),
                  num_classes=num_classes,
                  learning_rate=learning_rate,
                  clients_num=num)


def run_global_test(client, global_vars, test_num):
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))

def upload_vars(client_vars_sum, current_client_vars):
    """
    Upload parameters
    """
    if client_vars_sum == None:
        client_vars_sum = current_client_vars
        return client_vars_sum
    vars_num = len(current_client_vars)
    for i in range(vars_num):
        client_vars_sum[i] += current_client_vars[i]
    return client_vars_sum


#### SOME TRAINING PARAMS ####
CLIENT_NUMBER = 5   # The ml_privacy_meter can't handle the scenario with too many participants.
CLIENT_RATIO_PER_ROUND = 1.00
epoch = 360
# epoch = 60   # during debugging
input_shape = (32, 32, 3)   # the type of image in cifar-100

#### CREATE CLIENT AND LOAD DATASET ####
client = buildClients(CLIENT_NUMBER)

#### BEGIN TRAINING ####
global_vars = client.get_client_vars()
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
        client.set_global_vars(global_vars)
        # Perform the craft
        # if ep == 2 and client_id == 2:
        #     client.craft(cid=client_id)
        # else:
        #     client.train_epoch(cid=client_id)
        client.train_epoch(cid=client_id)

        # A passive inference attack can be performed here after crafting
        # if ep == 4 and client_id == client.craft_id:
        #     train_data = client.dataset.train[client_id]
        #     test_data = client.dataset.test
        #
        #     data_handler = ml_privacy_meter.utils.attack_data_v2.attack_data(test_data=test_data,
        #                                                                   train_data=train_data,
        #                                                                   batch_size=32,
        #                                                                   attack_percentage=10, input_shape=input_shape)
        #     cprefix = "../membership_inference_attack/tutorials/models/alexnet_pretrained"
        #     cmodelA = tf.keras.models.load_model(cprefix)
        #     cmodelA.summary()
        #     # client_model = client.model_object
        #     # client_model.summarry()
        #     data_handler.means, data_handler.stddevs = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        #     attackobj = ml_privacy_meter.attack.meminf_v2.initialize(
        #         target_train_model=cmodelA,
        #         target_attack_model=cmodelA,
        #         train_datahandler=data_handler,
        #         attack_datahandler=data_handler,
        #         layers_to_exploit=[26],
        #         # gradients_to_exploit=[6],
        #         device=None, epochs=10, model_name='blackbox1')
        #     attackobj.train_attack()
        #     attackobj.test_attack()

        # Cumulative updates
        current_client_vars = client.get_client_vars()
        if client_vars_sum is None:
            client_vars_sum = current_client_vars
        else:
            for cv, ccv in zip(client_vars_sum, current_client_vars):
                cv.assign_add(ccv)

    # obtain the avg vars as global vars
    size = len(client_vars_sum)
    for i in range(size):
        global_vars[i] = client_vars_sum[i] / len(random_clients)

    # run test on 1000 instances
    # run_global_test(client, global_vars, test_num=600)


#### FINAL TEST ####
# run_global_test(client, global_vars, test_num=1000)
