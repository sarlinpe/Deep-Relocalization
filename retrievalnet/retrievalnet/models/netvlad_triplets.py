import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v1 as resnet
from .utils import triplet_loss


def normalize_image(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    return tf.div(tf.subtract(image, pixel_value_offset), pixel_value_scale)


def netvlad(image, mode, config):
    image = normalize_image(image)
    if image.shape[-1] == 1:
        image = tf.tile(image, [1, 1, 1, 3])

    with slim.arg_scope(resnet.resnet_arg_scope()):
        training = config['train_backbone'] and (mode == Mode.TRAIN)
        with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=training):
            _, encoder = resnet.resnet_v1_50(image,
                                             is_training=training,
                                             global_pool=False,
                                             scope='resnet_v1_50')
    feature_map = encoder['resnet_v1_50/block3']

    with tf.variable_scope('vlad'):
        training = config['train_vlad'] and (mode == Mode.TRAIN)
        if config['intermediate_proj']:
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=training):
                with slim.arg_scope([slim.batch_norm], is_training=training):
                    feature_map = slim.conv2d(
                            feature_map, config['intermediate_proj'], 1, rate=1,
                            activation_fn=None, normalizer_fn=slim.batch_norm,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            trainable=training, scope='pre_proj')

        batch_size = tf.shape(feature_map)[0]
        feature_dim = feature_map.shape[-1]

        with slim.arg_scope([slim.batch_norm], trainable=training, is_training=training):
            memberships = slim.conv2d(
                    feature_map, config['n_clusters'], 1, rate=1,
                    activation_fn=None, normalizer_fn=slim.batch_norm,
                    weights_initializer=slim.initializers.xavier_initializer(),
                    trainable=training, scope='memberships')
            memberships = tf.nn.softmax(memberships, axis=-1)

        clusters = slim.model_variable(
                'clusters', shape=[1, 1, 1, config['n_clusters'], feature_dim],
                initializer=slim.initializers.xavier_initializer(), trainable=training)
        residuals = clusters - tf.expand_dims(feature_map, axis=3)
        residuals *= tf.expand_dims(memberships, axis=-1)
        descriptor = tf.reduce_sum(residuals, axis=[1, 2])

        descriptor = tf.nn.l2_normalize(descriptor, axis=1)  # intra-normalization
        descriptor = tf.reshape(descriptor,
                                [batch_size, feature_dim*config['n_clusters']])
        descriptor = tf.nn.l2_normalize(descriptor, axis=1)

    if config['dimensionality_reduction']:
        descriptor = tf.nn.l2_normalize(descriptor, -1)
        reg = slim.l2_regularizer(config['proj_regularizer']) \
            if config['proj_regularizer'] else None
        descriptor = slim.fully_connected(
                descriptor,
                config['dimensionality_reduction'],
                activation_fn=None,
                weights_initializer=slim.initializers.xavier_initializer(),
                trainable=True,
                weights_regularizer=reg,
                scope='dimensionality_reduction')
        descriptor = tf.nn.l2_normalize(descriptor, -1)

    return descriptor


class NetvladTriplets(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
            'p': {'shape': [None, None, None, 1], 'type': tf.float32},
            'n': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'triplet_margin': 0.5,
            'dimensionality_reduction': None,
            'intermediate_proj': None,
            'n_clusters': 64,
            'proj_regularizer': 0.,
            'train_backbone': False,
            'train_vlad': True,
            'loss_in': False,
            'loss_squared': True,
    }

    def _model(self, inputs, mode, **config):
        if mode == Mode.PRED:
            descriptor = netvlad(inputs['image'], mode, config)
            return {'descriptor': descriptor}

        descriptors = {}
        for e in ['image', 'p', 'n']:
            with tf.name_scope('triplet_{}'.format(e)):
                descriptors['descriptor_'+e] = netvlad(inputs[e], mode, config)
        return descriptors

    def _loss(self, outputs, inputs, **config):
        loss, _, _ = triplet_loss(outputs, inputs, **config)
        return loss

    def _metrics(self, outputs, inputs, **config):
        loss, distance_p, distance_n = triplet_loss(outputs, inputs, **config)
        return {'loss': loss, 'distance_p': distance_p, 'distance_n': distance_n}
