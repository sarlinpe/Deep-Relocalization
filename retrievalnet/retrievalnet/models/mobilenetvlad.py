import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import mobilenet_v2 as mobilenet


def normalize_image(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    return tf.div(tf.subtract(image, pixel_value_offset), pixel_value_scale)


def mobilenetvlad(image, mode, config):
    image = normalize_image(image)
    if image.shape[-1] == 1:
        image = tf.tile(image, [1, 1, 1, 3])
    if config['resize_input']:
        new_size = tf.to_int32(tf.round(
                tf.to_float(tf.shape(image)[1:3]) / float(config['resize_input'])))
        image = tf.image.resize_images(image, new_size)

    is_training = config['train_backbone'] and (mode == Mode.TRAIN)
    with slim.arg_scope(mobilenet.training_scope(
            is_training=is_training, dropout_keep_prob=config['dropout_keep_prob'])):
        _, encoder = mobilenet.mobilenet(image, num_classes=None, base_only=True,
                                         depth_multiplier=config['depth_multiplier'])
    feature_map = encoder['layer_18']

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
        descriptor = slim.fully_connected(
                descriptor,
                config['dimensionality_reduction'],
                activation_fn=None,
                weights_initializer=slim.initializers.xavier_initializer(),
                trainable=True,
                scope='dimensionality_reduction')
        descriptor = tf.nn.l2_normalize(descriptor, -1)

    return descriptor


def l2_error(inputs, outputs):
    return tf.reduce_sum(tf.square(inputs['descriptor'] - outputs['descriptor']),
                         axis=-1)/2


class Mobilenetvlad(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'depth_multiplier': 1.0,
            'resize_input': False,
            'dropout_keep_prob': None,
            'dimensionality_reduction': None,
            'intermediate_proj': None,
            'n_clusters': 64,
            'proj_regularizer': 0.,
            'train_backbone': True,
            'train_vlad': True,
    }

    def _model(self, inputs, mode, **config):
        descriptor = mobilenetvlad(inputs['image'], mode, config)
        return {'descriptor': descriptor}

    def _loss(self, outputs, inputs, **config):
        return tf.reduce_mean(l2_error(inputs, outputs))

    def _metrics(self, outputs, inputs, **config):
        return {'l2_error': l2_error(inputs, outputs)}
