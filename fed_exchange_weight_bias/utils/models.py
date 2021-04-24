import tensorflow as tf
import tensorflow.keras as keras


def alexnet(input_shape, classes_num=100):
    """
    AlexNet:
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    """
    # Creating initializer, optimizer and the regularizer ops
    initializer = keras.initializers.RandomNormal(0.0, 0.01)
    regularizer = keras.regularizers.l2(5e-4)

    if input_shape[2] == 1:
        input_shape = (input_shape[0], input_shape[1], input_shape[2],)
    else:
        input_shape = (input_shape[0], input_shape[1], input_shape[2],)

    # Creating the model
    model = tf.compat.v1.keras.Sequential(
        [
            keras.layers.Conv2D(
                64, 11, 4,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                input_shape=input_shape,
                data_format='channels_last'
            ),
            keras.layers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keras.layers.Conv2D(
                192, 5,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keras.layers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keras.layers.Conv2D(
                384, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keras.layers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keras.layers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            keras.layers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(
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


def create_model(model_name, input_shape, classes_num):
    if model_name == "alexnet":
        model = alexnet(input_shape, classes_num)
    elif model_name == "vgg16":
        # https://keras.io/zh/applications/
        base_model = keras.applications.vgg16.VGG16(include_top=False, weights=None,
                                               input_shape=input_shape)
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        predictions = keras.layers.Dense(classes_num, activation="softmax")(x)

        model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    elif model_name == "vgg19":
        base_model = keras.applications.vgg19.VGG19(include_top=False, weights=None,
                                                    input_shape=input_shape)
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        predictions = keras.layers.Dense(classes_num, activation="softmax")(x)

        model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    else:
        raise Exception("No such model: {}".format(model_name))

    return model
