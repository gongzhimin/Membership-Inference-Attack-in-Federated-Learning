import tensorflow as tf

def cross_entropy_loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,  labels=labels)

    return loss