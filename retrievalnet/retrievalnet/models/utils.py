import tensorflow as tf


def triplet_loss(self, outputs, inputs, **config):
    distance_p = tf.norm(outputs['descriptor_image'] - outputs['descriptor_p'], axis=1)
    distance_n = tf.norm(outputs['descriptor_image'] - outputs['descriptor_n'], axis=1)
    if config['loss_in']:
        loss = tf.maximum(distance_p + config['triplet_margin'] - distance_n, 0)
        if config['loss_squared']:
            loss = tf.square(loss)
    else:
        dp = tf.square(distance_p) if config['loss_squared'] else distance_p
        dn = tf.square(distance_n) if config['loss_squared'] else distance_n
        loss = dp + tf.maximum(config['triplet_margin'] - dn, 0)
    return [tf.reduce_mean(i) for i in [loss, distance_p, distance_n]]
