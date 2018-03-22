import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v1 as resnet


def normalize_image(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    return tf.div(tf.subtract(image, pixel_value_offset), pixel_value_scale)


class PoolingNet(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 3], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {'normalize': False}

    def _model(self, inputs, mode, **config):
        image = inputs['image']
        if config['normalize']:
            image = normalize_image(image)

        with slim.arg_scope(resnet.resnet_arg_scope()):
            _, encoder = resnet.resnet_v1_50(image,
                                             is_training=(mode == Mode.TRAIN),
                                             global_pool=False,
                                             scope='resnet_v1_50')
        feature_map = encoder['resnet_v1_50/block3']

        descriptor = tf.reduce_max(feature_map, [1, 2])
        return {'descriptor': descriptor}

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
