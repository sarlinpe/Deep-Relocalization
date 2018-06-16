import numpy as np
import tensorflow as tf
import cv2
import glob
import random
from pathlib import Path

from .base_dataset import BaseDataset


class DescriptorDistillation(BaseDataset):
    default_config = {
        'validation_size': 200,
        'truncate': None,
        'image_folders': [],
        'descriptor_folders': [],
        'cache_in_memory': False,
        'preprocessing': {
            'resize': [480, 640],
        }
    }

    def _init_dataset(self, **config):
        assert len(config['image_folders']) > 0
        assert len(config['image_folders']) == len(config['descriptor_folders'])

        image_paths = []
        descriptor_paths = []
        for i_folder, d_folder in zip(config['image_folders'],
                                      config['descriptor_folders']):
            i_paths = sorted(glob.glob(Path(i_folder, '*.jpg').as_posix()))
            names = [Path(p).stem for p in i_paths]
            d_paths = [Path(d_folder, '{}.npy'.format(n)).as_posix() for n in names]

            image_paths.extend(i_paths)
            descriptor_paths.extend(d_paths)
        paths = list(zip(image_paths, descriptor_paths))

        # Shuffle
        random.Random(0).shuffle(paths)
        if config['truncate'] is not None:
            paths = paths[:config['truncate']]
        return paths

    def _get_data(self, paths, split_name, **config):
        def _py_read_data(i_path, d_path):
            image = cv2.imread(i_path.decode('utf-8'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
            descriptor = np.load(d_path.decode('utf-8')).astype(np.float32)
            return image, descriptor

        def _preprocess(image, descriptor):
            image = tf.image.rgb_to_grayscale(image)
            image.set_shape([None, None, 1])
            image = tf.image.resize_images(image, config['preprocessing']['resize'],
                                           method=tf.image.ResizeMethod.BILINEAR)
            return image, descriptor

        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(
                lambda d: tuple(tf.py_func(
                    _py_read_data, [d[0], d[1]], [tf.uint8, tf.float32])),
                num_parallel_calls=10)
        dataset = dataset.map(_preprocess, num_parallel_calls=10)
        dataset = dataset.map(lambda i, d: {'image': i, 'descriptor': d})

        dataset = dataset.skip(config['validation_size'])
        if split_name == 'validation':
            dataset = dataset.take(config['validation_size'])

        if config['cache_in_memory']:
            tf.logging.info('Caching dataset, fist access will take some time.')
            dataset = dataset.cache()

        return dataset
