# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
def vgg_base(img_batch, scope_name, is_training=True):
  with tf.variable_scope(scope_name):
    net = slim.repeat(img_batch, 2, slim.conv2d, 64, [3, 3],
                      trainable=False, scope='conv1')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                      trainable=False, scope='conv2')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                      trainable=is_training, scope='conv3')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                      trainable=is_training, scope='conv4')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                      trainable=is_training, scope='conv5')

    return net

def vgg_head(inputs,scope_name, is_training):
  with tf.variable_scope(scope_name):
    pool5_flat = slim.flatten(inputs, scope='flatten')
    fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
    if is_training:
      fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                         scope='dropout6')
    fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
    if is_training:
      fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                         scope='dropout7')
    return fc7

