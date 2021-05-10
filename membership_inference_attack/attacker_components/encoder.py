import tensorflow as tf


keras_layers = tf.keras.layers


def create_encoder(encoder_inputs):
    initializer = tf.keras.initializers.RandomNormal(0.0, 0.01)
    inputs = keras_layers.concatenate(encoder_inputs, axis=1)

    encoder = keras_layers.Dense(
        256,
        input_shape=(int(inputs.shape[1]),),
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        bias_initializer="zeros"
    )(inputs)
    encoder = keras_layers.Dense(
        128,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        bias_initializer="zeros" 
    )(encoder)
    encoder = keras_layers.Dense(
        1,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        bias_initializer="zeros"
    )(encoder)

    return encoder
