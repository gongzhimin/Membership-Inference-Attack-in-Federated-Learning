import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.train import GradientDescentOptimizer
from tensorflow.compat.v1.train import AdamOptimizer
keras = tf.keras
from math import pi
keraslayers = tf.compat.v1.keras.layers
import math
from collections import namedtuple
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
input_shape = (32, 32, 3)
def AlexNet(input_shape, num_classes, learning_rate, graph):#构建模型
    with graph.as_default():
        X = tf.compat.v1.placeholder(tf.float32, input_shape, name='X')#输入
        Y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name='Y')#输出
        DROP_RATE = tf.compat.v1.placeholder(tf.float32, name='drop_rate')

        # Creating the model
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        # conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1 = conv(X, 11, 11, 96, 2, 2, name='conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        # flattened = tf.reshape(pool5, [-1, 6*6*256])
        # fc6 = fc(flattened, 6*6*256, 4096, name='fc6')

        flattened = tf.reshape(pool5, [-1, 1 * 1 * 256])
        fc6 = fc_layer(flattened, 1 * 1 * 256, 1024, name='fc6')
        dropout6 = dropout(fc6, DROP_RATE)

        # 7th Layer: FC (w ReLu) -> Dropout
        # fc7 = fc(dropout6, 4096, 4096, name='fc7')
        fc7 = fc_layer(dropout6, 1024, 2048, name='fc7')
        dropout7 = dropout(fc7, DROP_RATE)

        # 8th Layer: FC and return unscaled activations
        logits = fc_layer(dropout7, 2048, num_classes, relu=False, name='fc8')

        # loss and optimizer
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=Y))
        optimizer = AdamOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)

        # accuracy
        correct_pred = tf.equal(pred, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32))
        
        grads = optimizer.compute_gradients(loss_op)

        return X, Y, DROP_RATE, train_op, loss_op, accuracy, prediction, grads


def conv(x, filter_height, filter_width, num_filters,
            stride_y, stride_x, name, padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(
        i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.compat.v1.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.compat.v1.get_variable('weights',
                                    shape=[
                                        filter_height, filter_width,
                                        int(input_channels / groups), num_filters
                                    ])
        biases = tf.compat.v1.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3,
                                    num_or_size_splits=groups,
                                    value=weights)
        output_groups = [
            convolve(i, k) for i, k in zip(input_groups, weight_groups)
        ]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc_layer(x, input_size, output_size, name, relu=True, k=20):
    """Create a fully connected layer."""

    with tf.compat.v1.variable_scope(name) as scope:
        # Create tf variables for the weights and biases.
        W = tf.compat.v1.get_variable('weights', shape=[input_size, output_size])
        b = tf.compat.v1.get_variable('biases', shape=[output_size])
        # Matrix multiply weights and inputs and add biases.
        z = tf.nn.bias_add(tf.matmul(x, W), b, name=scope.name)

    if relu:
        # Apply ReLu non linearity.
        a = tf.nn.relu(z)
        return a

    else:
        return z

def max_pool(x,
                filter_height, filter_width,
                stride_y, stride_x,
                name, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool2d(x,
        ksize=[1, filter_height, filter_width, 1],
        strides=[1, stride_y, stride_x, 1],
        padding=padding,
        name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x,
        depth_radius=radius,
        alpha=alpha,
        beta=beta,
        bias=bias,
        name=name)


def dropout(x, rate):
    """Create a dropout layer."""
    return tf.nn.dropout(x, rate=rate)


FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op prediction grads')

class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)
        self.random_order = list(range(len(x)))
        np.random.shuffle(self.random_order)
        self.start = 0
        return
    def next_batch(self, batch_size):
        if self.start + batch_size >= len(self.random_order):
            overflow = (self.start + batch_size) - len(self.random_order)
            perm0 = self.random_order[self.start:] +\
                 self.random_order[:overflow]
            self.start = overflow
        else:
            perm0 = self.random_order[self.start:self.start + batch_size]
            self.start += batch_size

        assert len(perm0) == batch_size

        return self.x[perm0], self.y[perm0]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]

def normalize(f, means, stddevs):#标准化
    """
    Normalizes data using means and stddevs
    """
    normalized = (f/255 - means) / stddevs
    return normalized

dataset_path = "../datasets/cifar100.txt"

def extract(filepath):#提取数据集
    """
    Extracts dataset from given filepath
    """
    with open(filepath, "r") as f:

        dataset = f.readlines()
    dataset = map(lambda i: i.strip('\n').split(';'), dataset)
    dataset = np.array(list(dataset))
    return dataset


def extract_one(filepath):  # 提取数据集
    """
    Extracts dataset from given filepath
    """
    with open(filepath, "r") as f:
        dataset = f.readlines()
    dataset = map(lambda i: i.strip('\n').split(';'), dataset)
    dataset = np.array(list(dataset))
    random=np.random.randint(0, 59999, 1)
    data = dataset[random]
    return data

def generate(dataset, input_shape):#解析每一条数据
    """
    Parses each record of the dataset and extracts
    the class (first column of the record) and the
    features. This assumes 'csv' form of data.
    """
    features, labels = dataset[:, :-1], dataset[:, -1]
    features = map(lambda y: np.array(list(map(lambda i: i.split(","), y))).flatten(),
                   features)

    features = np.array(list(features))
    features = np.ndarray.astype(features, np.float32)

    if input_shape:
        if len(input_shape) == 3:
            reshape_input = (
                len(features),) + (input_shape[2], input_shape[0], input_shape[1])
            features = np.transpose(np.reshape(
                features, reshape_input), (0, 2, 3, 1))
        else:
            reshape_input = (len(features),) + input_shape
            features = np.reshape(features, reshape_input)

    labels = np.ndarray.astype(labels, np.float32)
    return features, labels


class Dataset(object):#数据集 训练数据、测试数据
    def __init__(self,split=0):
        # (x_train, y_train), (x_test, y_test) = load_data_func()
        dataset = extract(dataset_path)
        np.random.shuffle(dataset)

        features, labels = generate(dataset, input_shape)#特征，标签

        #opt = keras.optimizers.Adam(learning_rate=0.0001)


        # cmodel.compile(loss='categorical_crossentropy',
        #                optimizer=opt,
        #                metrics=['accuracy'])

        size = len(features)#每一条数据特征的长度
        num_classes = 100

        features_train = features[:int(0.8 * size)]#用来训练的特征40000
        features_test = features[int(0.8 * size):]#用来测试的特征10000

        features_train = normalize(features_train, [
            0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        features_test = normalize(features_test, [
            0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        labels_train = labels[:int(0.8* size)]
        labels_test = labels[int(0.8 * size):]

        labels_train = keras.utils.to_categorical(labels_train, num_classes)
        labels_test = keras.utils.to_categorical(labels_test, num_classes)
        self.train_labels=labels_train
        self.test_labels = labels_test
        self.train_features=features_train
        self.test_features=features_test
        if split == 0:
            self.train = BatchGenerator(features_train, labels_train)
        else:
            self.train = self.splited_batch(features_train, labels_train, split)

        self.test = BatchGenerator(features_test, labels_test)

    def splited_batch(self, x_data, y_data, count):
        res = []
        l = len(x_data)
        for i in range(0, l, l//count):
            res.append(
                BatchGenerator(x_data[i:i + l // count],
                               y_data[i:i + l // count]))
        return res
total_inputarray=[]
total_ninputarray=[]
class Clients:#创建参与者
    def __init__(self, input_shape, num_classes, learning_rate, clients_num):

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)

        # Call the create function to build the computational graph of AlexNet
        net = AlexNet(input_shape, num_classes, learning_rate, self.graph)

        self.model = FedModel(*net)
        # initialize
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())

        # Load Cifar-10 dataset
        # NOTE: len(self.dataset.train) == clients_num
        self.dataset = Dataset(split=clients_num)

    def run_test(self, num):#测试
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test.next_batch(num)
            feed_dict = {
                self.model.X: batch_x,
                self.model.Y: batch_y,
                self.model.DROP_RATE: 0
            }
        return self.sess.run([self.model.acc_op, self.model.loss_op],
                             feed_dict=feed_dict)

    def train_epoch(self, cid, batch_size=32, dropout_rate=0):#训练
        """
            Train one client with its own data for one epoch
            cid: Client id
        """
        dataset = self.dataset.train[cid]
        with self.graph.as_default():
            for _ in range(math.ceil(dataset.size // batch_size)):
                batch_x, batch_y = dataset.next_batch(batch_size)
                feed_dict = {
                    self.model.X: batch_x,
                    self.model.Y: batch_y,
                    self.model.DROP_RATE: dropout_rate
                }

                self.sess.run(self.model.train_op, feed_dict=feed_dict)
                # p= self.sess.run(self.model.pred, feed_dict=feed_dict)

    def train_attack(self, cid, batch_size=32, dropout_rate=0):  # 训练
        """
            Train one client with its own data for one epoch
            cid: Client id
        """

        features_train=self.dataset.train_features[0:1000] # 先选一条数据
        # print("features训练集的大小！！！！！！！！！！",len(features_train))
        features_test = self.dataset.test_features[0:1000]
        labels_train=self.dataset.train_labels[0:1000]
        # print("labels训练集的大小！！！！！！！！！！", len(labels_train))
        labels_test=self.dataset.test_labels[0:1000]
        zipped=zip(features_train,labels_train)
        zipped1=zip(features_test,labels_test)

        with self.graph.as_default():
            for (feature,label) in zipped:
                inputarray = []
                # print("feature的shape!!!!!!!!",feature.shape)
                # print("feature的类型!!!!!!!!", type(feature))
                # print("label的shape!!!!!!!!", label.shape)
                # print("label的类型!!!!!!!!", type(label))
                feature=feature.reshape([-1,32,32,3])
                # print("feature的shape!!!!!!!!",feature.shape)
                label = label.reshape([-1, 100])
                # print("label的shape!!!!!!!!", label.shape)
                #batch_x, batch_y = dataset.next_batch(batch_size)
                feed_dict = {
                    self.model.X: feature,
                    self.model.Y: label,
                    self.model.DROP_RATE: dropout_rate
                }
                # print("shuchu!!!!!!!!!!!",self.model.pred.shape)
                # print("label!!!!!!!!!!!", self.model.Y.shape)
                # print("loss!!!!!!!!!!!", self.model.loss_op.shape)
                p = self.sess.run(self.model.prediction, feed_dict=feed_dict)
                #p = keras.utils.to_categorical(p, 100)

                p1 = self.sess.run(self.model.Y, feed_dict=feed_dict)
                p2 = self.sess.run(self.model.loss_op, feed_dict=feed_dict)

                #p = tf.convert_to_tensor(p)
                #p1=tf.convert_to_tensor(self.model.loss_op)
                #p2= tf.reshape(p2, (len(p2.numpy()), 1))
                # p3 = self.sess.run(self.model.grads, feed_dict=feed_dict)
                # p4=self.model.X
                p2 = p2.reshape((1,1))
                inputarray.append(p)
                inputarray.append(p1)
                inputarray.append(p2)
                total_inputarray.append((inputarray))
                # inputarray.append(p3)
                # inputarray.append(p4)
                #print("inputarray:\n",inputarray)
                #print("p\n{}\np1{}\np2{}\n".format(p,p1,p2))
            #print("total_inputarray:\n", total_inputarray)
        with self.graph.as_default():
            for (nfeature, nlabel) in zipped1:
                ninputarray = []
                nfeature = nfeature.reshape([-1, 32, 32, 3])
                nlabel = nlabel.reshape([-1, 100])
                feed_dict = {
                    self.model.X: nfeature,
                    self.model.Y: nlabel,
                    self.model.DROP_RATE: dropout_rate
                }
                p = self.sess.run(self.model.prediction, feed_dict=feed_dict)
                p1 = self.sess.run(self.model.Y, feed_dict=feed_dict)
                p2 = self.sess.run(self.model.loss_op, feed_dict=feed_dict)
                p2 = p2.reshape((1, 1))
                ninputarray.append(p)
                ninputarray.append(p1)
                ninputarray.append(p2)
                total_ninputarray.append(ninputarray)
                #print("niputarray:\n",ninputarray)
                #print("total_niputarray:\n",total_ninputarray)
        return total_inputarray,total_ninputarray
    def get_client_vars(self):
        """ Return all of the variables list """

        with self.graph.as_default():
            client_vars = self.sess.run(tf.compat.v1.trainable_variables())
        return client_vars

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        with self.graph.as_default():
            all_vars = tf.compat.v1.trainable_variables()
            for variable, value in zip(all_vars, global_vars):
                variable.load(value, self.sess)

    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        client_num = self.get_clients_num()
        choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        return len(self.dataset.train)

# def forward_pass(self, features, labels,num_classes,learning_rate):  # 前向传播
#     """
#     Computes and collects necessary inputs for attack model
#     """
#     # container to extract and collect inputs from target model#从目标模型获得的输入来训练攻击模型
#     self.inputArray = []
#     net1 = AlexNet(input_shape, num_classes, learning_rate, self.graph)
#     model = FedModel(*net1)
#
#     with self.graph.as_default():
#
#         feed_dict = {
#             self.model.X: features,
#             self.model.Y: labels,
#             self.model.DROP_RATE: 0
#         }
#
#     predicted = self.sess.run(self.model.pred,feed_dict=feed_dict)
#     inputarray.append(predicted)
#     loss=self.model.loss_op
#     loss = tf.reshape(loss, (len(loss.numpy()), 1))
#     inputarray.append(loss)
#     inputarray.append(labels)
#
#
#     # Getting the gradients
#
#     # get_gradients(self.model, features, labels)  # 梯度
#
#     # attack_outputs = self.attackmodel(self.inputArray)  # 攻击模型的输出
#     return inputarray
#'''
# def attack_FL(inputarrary):
#     attackobj.train_attack()


def buildClients(num):#创建参与者
    learning_rate = 0.0001
    num_input = 32  # image shape: 32*32
    num_input_channel = 3  # image channel: 3
    #input_shape = (32, 32, 3)
    num_classes = 100  # Cifar-10 total classes (0-9 digits)

    #create Client and model
    return Clients(input_shape=[None, num_input, num_input, num_input_channel],
                  num_classes=num_classes,
                  learning_rate=learning_rate,
                  clients_num=num)





def getAttacktrain():
    CLIENT_NUMBER = 6 # 5个参与者
    CLIENT_RATIO_PER_ROUND = 0.12
    epoch = 25

    #### CREATE CLIENT AND LOAD DATASET ####
    client = buildClients(CLIENT_NUMBER)

    #### BEGIN TRAINING ####
    global_vars = client.get_client_vars()
    for ep in range(epoch):
        # We are going to sum up active clients' vars at each epoch
        client_vars_sum = None  # 把每一轮参与者的变量加起来

        # Choose some clients that will train on this epoch
        # random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)#随机选一些参与者
        random_clients = [1, 2, 3, 4, 5]

        # Train with these clients
        for client_id in tqdm(random_clients, ascii=True):
            # Restore global vars to client's model
            client.set_global_vars(global_vars)  # 参与者只能看到这里的全局变量

            # train one client
            client.train_epoch(cid=client_id)  # 参与者每一轮的训练
            if (client_id == 1):  # 这个攻击者要从数据集中选一个数据
                if (ep == 24):
                    print("执行了getFL\n")
                    print(client.train_attack(cid=1))





def run_global_test(client, global_vars, test_num):#全局模型的测试
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))

if __name__ == '__main__':
    #### SOME TRAINING PARAMS ####
    CLIENT_NUMBER = 6#5个参与者
    CLIENT_RATIO_PER_ROUND = 0.12
    epoch = 30

    #### CREATE CLIENT AND LOAD DATASET ####
    client = buildClients(CLIENT_NUMBER)

    #### BEGIN TRAINING ####
    global_vars = client.get_client_vars()
    for ep in range(epoch):
        # We are going to sum up active clients' vars at each epoch
        client_vars_sum = None#把每一轮参与者的变量加起来

        # Choose some clients that will train on this epoch
        #random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)#随机选一些参与者
        random_clients=[1,2,3,4,5]
        #print(random_clients)
        # Train with these clients
        for client_id in tqdm(random_clients, ascii=True):
            # Restore global vars to client's model
            client.set_global_vars(global_vars)#参与者只能看到这里的全局变量

            # train one client
            client.train_epoch(cid=client_id)#参与者每一轮的训练
            if(client_id==1):#这个攻击者要从数据集中选一个数据
                if(ep==29):
                    client.train_attack(cid=1)
                #获取全局模型


            # obtain current client's vars
            current_client_vars = client.get_client_vars()#参与者现在的变量

            # sum it up
            if client_vars_sum is None:
                client_vars_sum = current_client_vars
            else:
                for cv, ccv in zip(client_vars_sum, current_client_vars):
                    cv += ccv

        # obtain the avg vars as global vars
        global_vars = []
        for var in client_vars_sum:
            global_vars.append(var / len(random_clients))#平均聚合

        # run test on 1000 instances
        run_global_test(client, global_vars, test_num=600)#用全局变量进行全局模型的测试
#每个参与者的变量加起来再除以人数

    #### FINAL TEST ####
    run_global_test(client, global_vars, test_num=1000)
    #最后来个测试

import ml_privacy_meter
import tensorflow as tf
input_shape = (32, 32, 3)
cprefix = 'alexnet_pretrained'
cmodelA = tf.keras.models.load_model(cprefix)
cmodelA.summary()
saved_path = "datasets/cifar100_train.txt.npy"
dataset_path = 'datasets/cifar100.txt'
datahandlerA = ml_privacy_meter.utils.attack_data.attack_data(dataset_path=dataset_path,
                                                              member_dataset_path=saved_path,
                                                              batch_size=1,
                                                              attack_percentage=10, input_shape=input_shape,
                                                              normalization=True)
datahandlerA.means, datahandlerA.stddevs = [
    0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
attackobj = ml_privacy_meter.attack.meminf3.initialize(wdi=total_inputarray,wdni=total_ninputarray,
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[26],
    #gradients_to_exploit=[6],
    device=None, epochs=10, model_name='blackbox1')
attackobj.train_attack()

