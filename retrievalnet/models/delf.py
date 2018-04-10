import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v1 as resnet


def normalize_image(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    return tf.div(tf.subtract(image, pixel_value_offset), pixel_value_scale)


class Delf(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, None], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'normalize_input': False,
            'use_attention': False,
            'attention_kernel': 1,
            'normalize_average': True,
            'normalize_feature_map': True
    }

    def _model(self, inputs, mode, **config):
        image = inputs['image']
        if image.shape[-1] == 1:
            image = tf.tile(image, [1, 1, 1, 3])
        if config['normalize_input']:
            image = normalize_image(image)

        with slim.arg_scope(resnet.resnet_arg_scope()):
            _, encoder = resnet.resnet_v1_50(image,
                                             is_training=(mode == Mode.TRAIN),
                                             global_pool=False,
                                             scope='resnet_v1_50')
        feature_map = encoder['resnet_v1_50/block3']

        if config['use_attention']:
            with tf.variable_scope('attonly/attention/compute'):
                with slim.arg_scope(resnet.resnet_arg_scope()):
                    with slim.arg_scope([slim.batch_norm],
                                        is_training=(mode == Mode.TRAIN)):
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

        return {'descriptor': descriptor}

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
