import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from .base_dataset import BaseDataset
from retrievalnet.settings import DATA_PATH
from retrievalnet.datasets.utils.nclt_undistort import Undistort


class Nclt(BaseDataset):
    default_config = {
        'validation_size': 200,
        'cache_in_memory': False,
        'camera': 4,
        'preprocessing': {
            'undistort': False,
            'grayscale': True,
            'resize': [640, 488],
        }
    }

    def _init_dataset(self, **config):
        base_path = Path(DATA_PATH, 'datasets/nclt')
        self.split_names = []
        paths = {}

        # Triplets for training
        if 'training_triplets' in config:
            self.split_names.extend(['training', 'validation'])
            training_triplets = np.load(Path(base_path, config['training_triplets']))
            if 'validation_triplets' in config:
                validation_triplets = np.load(
                        Path(base_path, config['validation_triplets']))
                validation_triplets = validation_triplets[:config['validation_size']]
            else:
                validation_triplets = training_triplets[:config['validation_size']]
                training_triplets = training_triplets[config['validation_size']:]
            splits = {'validation': validation_triplets, 'training': training_triplets}
            for s in splits:
                paths[s] = {'image': [], 'p': [], 'n': []}
                for triplet in splits[s]:
                    for (seq, time), e in zip(triplet, ['image', 'p', 'n']):
                        paths[s][e].append(
                                str(Path(base_path, '{}/lb3/Cam{}/{}.tiff'.format(
                                    seq, config['camera'], time))))

        # Images for testing
        if 'test_seq' in config:
            self.split_names.append('test')
            paths['test'] = {}
            seq_file = np.loadtxt(
                    Path(base_path, 'pose_{}.csv'.format(config['test_seq'])),
                    dtype={'names': ('time', 'pose_x', 'pose_y', 'pose_angle'),
                           'formats': (np.int, np.float, np.float, np.float)},
                    delimiter=',', skiprows=1)
            paths['test']['name'] = seq_file['time'].astype(str).tolist()
            paths['test']['image'] = [str(Path(base_path, '{}/lb3/Cam{}/{}.tiff'.format(
                    config['test_seq'], config['camera'], n)))
                    for n in paths['test']['name']]

        return paths

    def _get_data(self, paths, split_name, **config):
        def _read_image(path):
            return cv2.imread(path.decode('utf-8'))

        def _undistort():
            undistort_file = Path(DATA_PATH, 'datasets/nclt/undistort_maps/',
                                  'U2D_Cam{}_1616X1232.txt'.format(config['camera']))
            undistort_map = Undistort(undistort_file)
            h, w = undistort_map.mask.shape
            x_min, x_max = [f(np.where(undistort_map.mask[int(h/2), :])[0])
                            for f in [np.min, np.max]]
            y_min, y_max = [f(np.where(undistort_map.mask[:, int(w/2)])[0])
                            for f in [np.min, np.max]]

            def _f(image):
                return undistort_map.undistort(image)[y_min:y_max, x_min:x_max, ...]
            return _f

        def _preprocess(image):
            image = tf.image.rot90(image, k=3)
            if config['preprocessing']['undistort']:
                image = tf.py_func(_undistort(), [image], tf.uint8)
            if config['preprocessing']['grayscale']:
                image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = tf.image.resize_images(image, config['preprocessing']['resize'],
                                               method=tf.image.ResizeMethod.BILINEAR)
            return image

        datasets = {}
        for e in paths[split_name]:
            d = tf.data.Dataset.from_tensor_slices(paths[split_name][e])
            if e != 'name':
                d = d.map(lambda path: tf.py_func(_read_image, [path], tf.uint8),
                          num_parallel_calls=13)
                d = d.map(_preprocess,
                          num_parallel_calls=13)
            datasets[e] = d
        dataset = tf.data.Dataset.zip(datasets)

        if config['cache_in_memory']:
            tf.logging.info('Caching dataset, fist access will take some time.')
            dataset = dataset.cache()

        return dataset
