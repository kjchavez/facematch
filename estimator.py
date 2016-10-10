from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.contrib.framework.python.ops import variables
from input import *

tf.logging.set_verbosity(tf.logging.INFO)

def conv_layer(name, shape, image):
    kernel = variables.model_variable('%s/W' % (name,), shape=shape,
                                      initializer=tf.random_normal_initializer(mean=0.0,
                                                                               stddev=0.1))

    b = variables.model_variable('%s/b' % (name,), shape=[shape[-1]],
                                 initializer=tf.zeros_initializer([shape[-1]]))

    embedding = tf.nn.bias_add(tf.nn.conv2d(image, kernel, [1, 1, 1, 1],
                                          padding='SAME'), b)
    embedding = tf.nn.max_pool(tf.nn.relu(embedding), [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    return embedding

def embed(image):
    """ Creates embedding tensors for the given batch of images. """
    embedding = conv_layer("conv1", [3, 3, 3, 16], image)
    embedding = conv_layer("conv2", [3, 3, 16, 32], embedding)


# The features for this model will always be a ?
def model_fn(features, targets, mode):
    """ Must return (predictions, loss, train_op). """

    # Create a set of variables.
    weights = variables.model_variable(
        'weights', shape=[250*250*3*2],
        initializer=tf.random_normal_initializer(mean=0.0,
                                                 stddev=0.1))

    image1 = features['image1']
    image2 = features['image2']
    label = tf.cast(targets, tf.float32)

    print(image1)
    print(image2)

    image1_flat = tf.reshape(image1, [-1, 250*250*3])
    image2_flat = tf.reshape(image2, [-1, 250*250*3])

    concat = tf.concat(1, [image1_flat, image2_flat])
    print(concat)

    logit = tf.reduce_mean(weights * concat, 1, keep_dims=True)
    print(logit)
    print(label)
    prediction = tf.nn.sigmoid(logit)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logit, label)
    print("Loss", loss)
    mean_loss = tf.reduce_mean(loss)

    # Compute a dummy loss, gradient, train_op
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    global_step = variables.get_global_step()
    train_op = optimizer.minimize(mean_loss, global_step=global_step)

    return (prediction, mean_loss, train_op)


estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir="/tmp/test")
estimator.fit(input_fn=lambda: get_train_data(batch_size=32), steps=1000)
print(estimator.evaluate(input_fn=lambda:
                         get_eval_data(batch_size=32),steps=16))
