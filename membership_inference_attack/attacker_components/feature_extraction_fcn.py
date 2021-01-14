import tensorflow as tf

keras_layers = tf.compat.v1.keras.layers

def create_fcn_component(input_size, layer_size=128):
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)

    fcn_component = tf.compat.v1.keras.Sequential(
        [
            keras_layers.Dense(
                layer_size,
                activation=tf.nn.relu,
                input_shape=(input_size, ),
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

    return fcn_component

