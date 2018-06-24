import tensorflow as tf
import numpy as np
import glob
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

import netvlad_tf.nets as nets


def read_and_preprocess(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_all_paths(folder):
    return sorted(glob.glob(Path(folder, '*.jpg').as_posix()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    jpeg_paths = get_all_paths(input_path)
    output_path.mkdir(exist_ok=True)
    batch_size = 16

    tf_batch = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3])
    net_out = nets.vgg16NetvladPca(tf_batch)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, nets.defaultCheckpoint())

    for batch_offset in tqdm(range(0, len(jpeg_paths), batch_size),
                             unit_scale=batch_size):
        images = []
        batch_paths = []
        for i in range(batch_offset, batch_offset + batch_size):
            if i == len(jpeg_paths):
                break
            images.append(read_and_preprocess(jpeg_paths[i]))
            batch_paths.append(jpeg_paths[i])
        batch = np.stack(images)
        descriptors = sess.run(net_out, feed_dict={tf_batch: batch})
        for i, p in enumerate(batch_paths):
            output_name = Path(p).stem
            np.save(Path(output_path, '{}.npy'.format(output_name)).as_posix(),
                    descriptors[i])
