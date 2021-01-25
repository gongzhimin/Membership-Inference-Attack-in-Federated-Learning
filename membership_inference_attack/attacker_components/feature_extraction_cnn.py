import tensorflow as tf

keras_layers = tf.compat.v1.keras.layers


def create_cnn_for_fcn_gradients(input_shape):
    dim0 = int(input_shape[0])
    dim1 = int(input_shape[1])
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)

    cnn_for_fcn_gradients = tf.compat.v1.keras.Sequential(
        [
            keras_layers.Dropout(0.2, input_shape=(dim0, dim1, 1, ), ),
            keras_layers.Conv2D(
                100,
                kernel_size=(1, dim1),
                strides=(1, 1),
                padding="valid",
                activation=tf.nn.relu,
                data_format="channels_last",
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Flatten(),
            keras_layers.Dropout(0.2),
            keras_layers.Dense(
                2024,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dropout(0.2, input_shape=(dim0, dim1, 1, ), ),
            keras_layers.Dense(
                512,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dense(
                256,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            )
        ]
    )

    return cnn_for_fcn_gradients


def create_cnn_for_cnn_layer_outputs(input_shape):
    dim0 = int(input_shape[1])
    dim1 = int(input_shape[2])
    dim2 = int(input_shape[3])
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)

    cnn_for_cnn_layer_output = tf.compat.v1.keras.Sequential(
        [
            keras_layers.Conv2D(
                dim2,
                kernel_size=(dim0, dim1),
                strides=(1, 1),
                padding="valid",
                activation=tf.nn.relu,
                data_format="channels_last",
                input_shape=(dim0, dim1, dim2, ),
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Flatten(),
            keras_layers.Dropout(0.2),
            keras_layers.Dense(
                1024,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dropout(0.2),
            keras_layers.Dense(
                512,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dense(
                128,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            )
        ]
    )

    return cnn_for_cnn_layer_output


def create_cnn_for_cnn_gradients(input_shape):
    dim0 = int(input_shape[3])
    dim1 = int(input_shape[0])
    dim2 = int(input_shape[1])
    dim3 = int(input_shape[2])
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)

    cnn_for_cnn_gradients = tf.compat.v1.keras.Sequential(
        [
            keras_layers.Conv2D(
                dim0,
                kernel_size=(dim1, dim2),
                strides=(1, 1),
                padding="same",
                activation=tf.nn.relu,
                input_shape=(dim0, dim1, dim3),
                kernel_initializer=initializer,
                bias_initializer="zeros",
                name="cnn_grad_layer"
            ),
            keras_layers.Flatten(name="flatten_layer"),
            keras_layers.Dropout(0.2),
            keras_layers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            )
        ]
    )

    return cnn_for_cnn_gradients


