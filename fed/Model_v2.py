import os
import numpy as np
import tensorflow as tf

keras = tf.keras
keraslayers = tf.compat.v1.keras.layers


def alexnet(input_shape):
    """
    AlexNet:
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    """
    # Creating initializer, optimizer and the regularizer ops
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
    regularizer = tf.keras.regularizers.l2(5e-4)

    inputshape = (input_shape[0], input_shape[1], input_shape[2],)

    # Creating the model
    model = tf.compat.v1.keras.Sequential(
        [
            keraslayers.Conv2D(
                64, 11, 4,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                input_shape=inputshape,
                data_format='channels_last'
            ),
            keraslayers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keraslayers.Conv2D(
                192, 5,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keraslayers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keraslayers.Conv2D(
                384, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keraslayers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keraslayers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keraslayers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keraslayers.Flatten(),
            keraslayers.Dropout(0.3),
            keraslayers.Dense(
                100,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.softmax
            )
        ]
    )
    return model


input_shape = (32, 32, 3)
alexnet_model = alexnet(input_shape)

# Get the datahandler ('tf.data.Dataset' instance)
dataset_path = "../datasets/cifar100.txt"


def scheduler(epoch):
    """
    Learning rate scheduler
    """
    lr = 0.0001
    if epoch > 25:
        lr = 0.00001
    elif epoch > 60:
        lr = 0.000001
    print('Using learning rate', lr)
    return lr


def generate(dataset, input_shape):
    """
    Parses each record of the dataset and extracts
    the class (first column of the record) and the
    features. This assumes 'csv' form of data.
    """
    features, labels = dataset[:, :-1], dataset[:, -1]
    # features, labels = features[:500], labels[:500]
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


def extract(filepath):
    """
    Extracts dataset from given filepath
    """
    with open(filepath, "r") as f:
        dataset = f.readlines()
    dataset = map(lambda i: i.strip('\n').split(';'), dataset)
    dataset = np.array(list(dataset))
    return dataset

def compute_moments(features, input_channels=3):
    """
    Computes means and standard deviation for 3 dimensional input for normalization.
    """
    means = np.zeros(input_channels, dtype=np.float32)
    stddevs = np.zeros(input_channels, dtype=np.float32)
    for i in range(input_channels):
        # very specific to 3-dimensional input
        pixels = features[:, :, :, i].ravel()
        means[i] = np.mean(pixels, dtype=np.float32)
        stddevs[i] = np.std(pixels, dtype=np.float32)
    means = list(map(lambda i: np.float32(i / 255), means))
    stddevs = list(map(lambda i: np.float32(i / 255), stddevs))
    return means, stddevs

def normalize(f):
    """
    Normalizes data using means and stddevs
    """
    means, stddevs = compute_moments(f)
    normalized = (np.divide(f, 255) - means) / stddevs
    return normalized


if __name__ == '__main__':
    training_size = 30000
    batch_size = 128

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    dataset = extract(dataset_path)
    np.random.shuffle(dataset)

    features, labels = generate(dataset, input_shape)

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    alexnet_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

    size = len(features)
    num_classes = 100

    features_train = features[int(0.2 * size):]
    features_test = features[:int(0.2 * size)]

    features_train = normalize(features_train)

    features_test = normalize(features_test)

    labels_train = labels[int(0.2 * size):]
    labels_test = labels[:int(0.2 * size)]

    labels_train = keras.utils.to_categorical(labels_train, num_classes)
    labels_test = keras.utils.to_categorical(labels_test, num_classes)

    alexnet_model.fit(features_train, labels_train,
                      batch_size=128,
                      epochs=100,
                      validation_data=(features_test, labels_test),
                      shuffle=True, callbacks=[callback])

    # model_path = os.path.join('cifar100_model')
    # cmodel.save(model_path)
    # print('Saved trained model at %s ' % model_path)
