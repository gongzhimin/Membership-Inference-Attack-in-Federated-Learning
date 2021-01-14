import tensorflow as tf


keras_layers = tf.compat.v1.keras.layers


def create_encoder(encoder_inputs):
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
    appended = keras_layers.concatenate(encoder_inputs, axis=1)

    encoder = tf.compat.v1.keras.Sequential(
        [
            keras_layers.Dropout(0.2),
            keras_layers.Dense(
                256,
                input_shape=(int(appended.shape[1]), ),
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dropout(0.2),
            keras_layers.Dense(
                128,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dropout(0.2),
            keras_layers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            ),
            keras_layers.Dense(
                1,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer="zeros"
            )
        ]
    )

    return encoder