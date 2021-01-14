import tensorflow as tf


def generate_optimizer(optimizer_name, learning_rate, logger, momentum=0.9, decay=0.0005):
    optimizer_name_lowercase = optimizer_name.lower()
    if optimizer_name_lowercase == "adam":
        return tf.optimizers.Adam(learning_rate)
    elif optimizer_name_lowercase == "sgd":
        return tf.optimizers.SGD(learning_rate)
    elif optimizer_name_lowercase == "momentum":
        return tf.optimizers.SGD(learning_rate, momentum, decay)
    elif optimizer_name_lowercase == "rmsprop":
        return tf.optimizers.RMSprop(learning_rate)
    elif optimizer_name_lowercase == "adagrad":
        return tf.optimizers.Adagrad(learning_rate)
    elif optimizer_name_lowercase == "adadelta":
        return tf.optimizers.Adadelta(learning_rate)
    else:
        logger.fatal("No such optimizer '{}', please check the spelling!".format(optimizer_name))
