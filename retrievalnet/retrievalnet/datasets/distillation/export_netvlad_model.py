import tensorflow as tf
from pathlib import Path

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets

export_dir = './netvlad_tf-model'

tf.reset_default_graph()

net_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='image')
net_in = tf.tile(net_in, [1, 1, 1, 3])

net_out = nets.vgg16NetvladPca(net_in)
net_out = tf.identity(net_out, name='descriptor')

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, nets.defaultCheckpoint())

tf.saved_model.simple_save(
    sess,
    export_dir,
    inputs={'image': net_in},
    outputs={'descriptor': net_out})
