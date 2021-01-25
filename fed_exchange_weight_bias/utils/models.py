import numpy as np
import tensorflow as tf

keraslayers = tf.compat.v1.keras.layers


def alexnet(input_shape, classes_num=100):
    """
    AlexNet:
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    """
    # Creating initializer, optimizer and the regularizer ops
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01, dtype=tf.float32)
    regularizer = tf.compat.v1.keras.regularizers.l2(5e-4)

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
                classes_num,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.softmax
            )
        ]
    )
    return model


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


def load_cifar10():
    (features_train, labels_train), (features_test, labels_test) = tf.compat.v1.keras.datasets.cifar10.load_data()
    features_train, labels_train = features_train[:100], labels_train[:100]
    features_test, labels_test = features_test[:20], labels_test[:20]
    return features_train, labels_train, features_test, labels_test



def train_model(model, train_dataset, test_dataset, epochs):
    for epoch in range(epochs):
        if epoch == 25:
            model.optimizer.lr.assign(model.optimizer.lr / 10)
        elif epoch == 60:
            model.optimizer.lr.assign(model.optimizer.lr / 10)





if __name__ == '__main__':
    # training_size = 30000
    # batch_size = 128

    input_shape = (32, 32, 3)
    classes_num = 10
    # Load data.
    features_train, labels_train, features_test, labels_test = load_cifar10()

    # Preprocess the data.
    features_train = normalize(features_train)
    features_test = normalize(features_test)
    labels_train = tf.compat.v1.keras.utils.to_categorical(labels_train, classes_num)
    labels_test = tf.compat.v1.keras.utils.to_categorical(labels_test, classes_num)

    # Train the model.
    alexnet_model = alexnet(input_shape, classes_num)
    opt = tf.compat.v1.keras.optimizers.Adam(learning_rate=0.0001)
    alexnet_model.compile(optimizer=opt,
                          loss=tf.compat.v1.keras.losses.CategoricalCrossentropy(),
                          metrics=[tf.compat.v1.keras.metrics.CategoricalAccuracy()])
    # alexnet_model.compile(loss='categorical_crossentropy',
    #                       optimizer=opt,
    #                       metrics=['accuracy'])
    callback = tf.compat.v1.keras.callbacks.LearningRateScheduler(scheduler)
    alexnet_model.fit(features_train, labels_train,
                      batch_size=128,
                      epochs=100,
                      validation_data=(features_test, labels_test),
                      shuffle=True, callbacks=[callback])
