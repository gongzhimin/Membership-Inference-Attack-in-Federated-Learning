import numpy as np
from tensorflow.keras.utils import to_categorical


def extract(filepath):
    """
    Extract dataset from given filepath.
    """
    with open(filepath, "r") as f:
        dataset = f.readlines()
    # dataset = map(lambda i: i.strip('\n').encode('utf-8').decode("utf-8").split(';'), dataset)
    dataset = map(lambda i: i.strip('\n').split(';'), dataset)
    dataset = np.array(list(dataset))
    return dataset

def generate(dataset, input_shape):
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

def compute_moments(features, input_channels=3):
    """
    Computes means and standard deviation for 3 dimensional input for normalization.
    """
    means = []
    stddevs = []
    for i in range(input_channels):
        # very specific to 3-dimensional input
        pixels = features[:, :, :, i].ravel()
        means.append(np.mean(pixels, dtype=np.float32))
        stddevs.append(np.std(pixels, dtype=np.float32))
    means = list(map(lambda i: np.float32(i / 255), means))
    stddevs = list(map(lambda i: np.float32(i / 255), stddevs))
    return means, stddevs

def normalize(features):
    """
    Normalizes data using means and stddevs
    """
    means, stddevs = compute_moments(features)
    normalized = (features/255 - means) / stddevs
    return normalized

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
            perm0 = self.random_order[self.start:] + \
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


class Dataset(object):
    """
    Load the dataset from a specific file.
    """
    # The dataset_path is ./ml_privacy_meter/datasets/cifar100.txt
    def __init__(self, dataset_path, input_shape, classes_num, split, one_hot):
        dataset = extract(dataset_path)
        np.random.shuffle(dataset)  # shuffle the dataset to ensure the data records of each participant iid
        features, labels = generate(dataset, input_shape)

        # features = {ndarray: (60000, 32, 32, 3)}, stored the images
        # labels = {ndarray: (60000,)}, the labels of corresponding images
        # Slice the features as well as labels to accelerate the execution during debugging, forget about accuracy
        features, labels = features[:5000], labels[:5000]

        # Split the dataset into two parts: train set, test set.
        size = len(features)  # get the size of dataset
        features_train, labels_train = features[:int(0.8*size)], labels[:int(0.8*size)]
        features_test, labels_test = features[int(0.8*size):], labels[int(0.8*size):]

        # Normalize the train features and test features.
        features_train = normalize(features_train)
        features_test = normalize(features_test)

        # Perform one-hot encoding.
        if one_hot:
            labels_train = to_categorical(labels_train, classes_num)
            labels_test = to_categorical(labels_test, classes_num)

        print("Dataset: train-%d, test-%d" % (len(features_train), len(features_test)))
        # x_train = np.expand_dims(x_train, -1)
        # x_test = np.expand_dims(x_test, -1)

        # Register
        self.features_train, self.labels_train = features_train, labels_train
        self.features_test, self.labels_test = features_test, labels_test

        # Create data batches for participants.
        self.train = self.splited_batch(features_train, labels_train, split)
        # Create the testing batch.
        self.test = BatchGenerator(features_test, labels_test)

    def splited_batch(self, x_data, y_data, count):
        res = []
        l = len(x_data)
        if count == 0:
            res.append(BatchGenerator(x_data, y_data))
            return res
        for i in range(0, l, l//count):
            res.append(
                BatchGenerator(x_data[i:i + l // count],
                               y_data[i:i + l // count]))
        return res