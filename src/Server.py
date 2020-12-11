import tensorflow as tf
import sys
sys.path.append("../membership_inference_attack/")
import ml_privacy_meter
from tqdm import tqdm

from Client import Clients

tf.compat.v1.disable_eager_execution()
def buildClients(num):
    learning_rate = 0.0001
    num_input = 32
    num_input_channel = 3
    num_classes = 100

    #create Client and model
    return Clients(input_shape=[None, num_input, num_input, num_input_channel],
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

#


#### SOME TRAINING PARAMS ####
CLIENT_NUMBER = 100
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
    for client_id in tqdm(random_clients, ascii=True):
        # In each epoch, clients download parameters from the server
        # and then train the local model to update their parameters
        client.set_global_vars(global_vars)
        # Perform the craft
        if ep == 2 and client_id == 2:
            client.craft(cid=client_id)
        else:
            client.train_epoch(cid=client_id)

        # A passive inference attack can be performed here after crafting
        if ep == 4 and client_id == client.craft_id:
            cmodel = client.model_object
            cmodel.summarry()
            datahandler = ml_privacy_meter.utils.attack_data.attack_data(dataset_path=dataset_path,
                                                                          member_dataset_path=saved_path,
                                                                          batch_size=32,
                                                                          attack_percentage=10, input_shape=input_shape,
                                                                          normalization=True)
            datahandler.means, datahandler.stddevs = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
            attackobj = ml_privacy_meter.attack.meminf.initialize(
                target_train_model=cmodel,
                target_attack_model=cmodel,
                train_datahandler=datahandler,
                attack_datahandler=datahandler,
                layers_to_exploit=[8],
                gradients_to_exploit=[6],
                device=None, epochs=10, model_name='blackbox1')
            attackobj.train_attack()
            attackobj.test_attack()

        # Cumulative updates
        current_client_vars = client.get_client_vars()
        if client_vars_sum is None:
            client_vars_sum = current_client_vars
        else:
            for cv, ccv in zip(client_vars_sum, current_client_vars):
                cv += ccv

    # obtain the avg vars as global vars
    size = len(client_vars_sum)
    for i in range(size):
        global_vars[i] = client_vars_sum[i] / len(random_clients)

    # run test on 1000 instances
    run_global_test(client, global_vars, test_num=600)


#### FINAL TEST ####
run_global_test(client, global_vars, test_num=1000)
