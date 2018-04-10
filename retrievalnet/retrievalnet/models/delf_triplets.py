import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v1 as resnet


def normalize_image(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    return tf.div(tf.subtract(image, pixel_value_offset), pixel_value_scale)


def delf_model(image, mode, config):
    image = normalize_image(image)
    if image.shape[-1] == 1:
        image = tf.tile(image, [1, 1, 1, 3])

    with slim.arg_scope(resnet.resnet_arg_scope()):
        is_training = config['train_backbone'] and (mode == Mode.TRAIN)
        with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
            _, encoder = resnet.resnet_v1_50(image,
                                             is_training=is_training,
                                             global_pool=False,
                                             scope='resnet_v1_50')
    feature_map = encoder['resnet_v1_50/block3']

    if config['use_attention']:
        with tf.variable_scope('attonly/attention/compute'):
            with slim.arg_scope(resnet.resnet_arg_scope()):
                is_training = config['train_attention'] and (mode == Mode.TRAIN)
                with slim.arg_scope([slim.conv2d, slim.batch_norm],
                                    trainable=is_training):
                    with slim.arg_scope([slim.batch_norm], is_training=is_training):
                        attention = slim.conv2d(
                                feature_map, 512, config['attention_kernel'], rate=1,
                                activation_fn=tf.nn.relu, scope='conv1')
                        attention = slim.conv2d(
                                attention, 1, config['attention_kernel'], rate=1,
                                activation_fn=None, normalizer_fn=None, scope='conv2')
                        attention = tf.nn.softplus(attention)
        if config['normalize_feature_map']:
            feature_map = tf.nn.l2_normalize(feature_map, -1)
        descriptor = tf.reduce_sum(feature_map*attention, axis=[1, 2])
        if config['normalize_average']:
            descriptor /= tf.reduce_sum(attention, axis=[1, 2])
    else:
        descriptor = tf.reduce_max(feature_map, [1, 2])

    if config['dimensionality_reduction']:
        descriptor = tf.nn.l2_normalize(descriptor, -1)
        descriptor = slim.fully_connected(
                descriptor,
                config['dimensionality_reduction'],
                activation_fn=None,
                weights_initializer=slim.initializers.xavier_initializer(),
                trainable=True,
                scope='dimensionality_reduction')
        descriptor = tf.nn.l2_normalize(descriptor, -1)

    return descriptor


class DelfTriplets(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
            'p': {'shape': [None, None, None, 1], 'type': tf.float32},
            'n': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'use_attention': True,
            'attention_kernel': 1,
            'normalize_average': True,
            'normalize_feature_map': True,
            'triplet_margin': 0.5,
            'dimensionality_reduction': None,
            'train_backbone': False,
            'train_attention': True,
    }

    def _model(self, inputs, mode, **config):
        if mode == Mode.PRED:
            descriptor = delf_model(inputs['image'], mode, config)
            return {'descriptor': descriptor}

        descriptors = {}
        for e in ['image', 'p', 'n']:
            with tf.name_scope('triplet_{}'.format(e)):
                descriptors['descriptor_'+e] = delf_model(inputs[e], mode, config)
        return descriptors

    def _loss(self, outputs, inputs, **config):
        distance_p = tf.reduce_sum(tf.square(outputs['descriptor_image']
                                             - outputs['descriptor_p']), axis=-1)
        distance_n = tf.reduce_sum(tf.square(outputs['descriptor_image']
                                             - outputs['descriptor_n']), axis=-1)
        loss = distance_p + tf.maximum(config['triplet_margin'] - distance_n, 0)
        loss = tf.reduce_mean(loss)
        return loss

    def _metrics(self, outputs, inputs, **config):
        distance_p = tf.reduce_sum(tf.square(outputs['descriptor_image']
                                             - outputs['descriptor_p']), axis=-1)
        distance_n = tf.reduce_sum(tf.square(outputs['descriptor_image']
                                             - outputs['descriptor_n']), axis=-1)
        loss = distance_p + tf.maximum(config['triplet_margin'] - distance_n, 0)
        loss = tf.reduce_mean(loss)
        return {'loss': loss,
                'distance_n': tf.reduce_mean(distance_n),
                'distance_p': tf.reduce_mean(distance_p)}
