import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v1 as resnet


class PoolingNet(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 3], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {'pooling_scales': [1, 2, 4]}

    def _model(self, inputs, mode, **config):
        with slim.arg_scope(resnet.resnet_arg_scope()):
            _, encoder = resnet.resnet_v1_50(inputs['image'],
                                             is_training=(mode == Mode.TRAIN),
                                             global_pool=False,
                                             scope='resnet_v1_50')
        features = encoder['resnet_v1_50/block3']
        descriptor = tf.reduce_max(features, [1, 2])
        return {'descriptor': descriptor}

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
