import tensorflow as tf

def cross_entropy_loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,  labels=labels)

    return loss

def mse(y_true, y_pred):
    loss = tf.losses.mean_squared_error(y_true, y_pred)

    return loss