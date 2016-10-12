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
    """ Adds a convolutional layer with a 3x3 filter, relu, and 2x2 max pool.
    """
    # What is the "ideal" initialization? Let's do N(0, 2/n). i.e. stddev =
    # sqrt(2/n) where n is the number of inputs to that layer. in this case, we
    # have n = 3 x 3 x depth of previous layer.
    N = shape[0]*shape[1]*shape[2]
    kernel = variables.model_variable('%s_W' % (name,), shape=shape,
                                      initializer=tf.random_normal_initializer(
                                          mean=0.0, stddev=np.sqrt(2.0/N)))


    b = variables.model_variable('%s_b' % (name,),
                                 initializer=tf.zeros_initializer([shape[-1]]))

    embedding = tf.nn.bias_add(tf.nn.conv2d(image, kernel, [1, 1, 1, 1],
                                          padding='SAME'), b)
    embedding = tf.nn.max_pool(tf.nn.relu(embedding), [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    return embedding

def embed(image):
    """ Creates embedding tensors for the given batch of images. """
    embedding = conv_layer("conv1", [3, 3, 3, 16], image)
    embedding = conv_layer("conv2", [3, 3, 16, 32], embedding)
    embedding = conv_layer("conv3", [3, 3, 32, 64], embedding)
    embedding = conv_layer("conv4", [3, 3, 64, 64], embedding)
    embedding = conv_layer("conv5", [3, 3, 64, 64], embedding)
    # At this point, we have shape = (BATCH, 4, 4, 64)
    # Do we want to insert fully connected layers as well? Yes.

    W = variables.model_variable("fc1_W", shape=(1024, 512),
                                  initializer=tf.random_normal_initializer(
                                      mean=0.0, stddev=1e-2))
    b = variables.model_variable("fc1_b",
                                 initializer=tf.zeros_initializer([512]))
    embedding = tf.nn.bias_add(tf.matmul(tf.reshape(embedding, (-1, 4*4*64)),
                                         W), b)
    return embedding


def resize_image(images, size):
    return tf.image.resize_images(images, size)

def model_fn(features, targets, mode):
    image1 = resize_image(features['image1'], (128, 128))
    image2 = resize_image(features['image2'], (128, 128))
    label = tf.cast(targets, tf.float32)

    with tf.variable_scope("embedding"):
        embedding1 = embed(image1)
        #embedding1 = tf.nn.l2_normalize(embedding1, 1)

    with tf.variable_scope("embedding", reuse=True):
        embedding2 = embed(image2)
        #embedding2 = tf.nn.l2_normalize(embedding2, 1)

    prod = embedding1*embedding2
    print("Product of embeddings:", prod)
    logit = tf.reduce_sum(prod, 1, keep_dims=True)
    tf.histogram_summary("logits", logit)

    prediction = tf.nn.sigmoid(logit)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logit, label)
    mean_loss = tf.reduce_mean(loss)

    # Compute a dummy loss, gradient, train_op
    optimizer = tf.train.RMSPropOptimizer(1e-4)
    global_step = variables.get_global_step()
    train_op = optimizer.minimize(mean_loss, global_step=global_step)

    return (prediction, mean_loss, train_op)

# The features for this model will always be a ?
def baseline_model_fn(features, targets, mode):
    """ Must return (predictions, loss, train_op). """

    # Create a set of variables.
    weights = variables.model_variable(
        'weights', shape=[128*128*3*2],
        initializer=tf.random_normal_initializer(mean=0.0,
                                                 stddev=0.1))

    image1 = resize_image(features['image1'], (128, 128))
    image2 = resize_image(features['image2'], (128, 128))
    label = tf.cast(targets, tf.float32)

    print(image1)
    print(image2)

    image1_flat = tf.reshape(image1, [-1, 128*128*3])
    image2_flat = tf.reshape(image2, [-1, 128*128*3])

    concat = tf.concat(1, [image1_flat, image2_flat])

    logit = tf.reduce_mean(weights * concat, 1, keep_dims=True)
    print(logit)
    print(label)
    prediction = tf.nn.sigmoid(logit)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logit, label)
    print("Loss", loss)
    mean_loss = tf.reduce_mean(loss)

    # Compute a dummy loss, gradient, train_op
    optimizer = tf.train.RMSPropOptimizer(0.001)
    global_step = variables.get_global_step()
    train_op = optimizer.minimize(mean_loss, global_step=global_step)

    return (prediction, mean_loss, train_op)


config = tf.contrib.learn.RunConfig(save_checkpoints_secs=60)
estimator = tf.contrib.learn.Estimator(model_fn=model_fn,
                                       model_dir="/tmp/test", config=config)

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=lambda: get_eval_data(batch_size=32), eval_steps=16,
    every_n_steps=1000)
grad_monitor = tf.contrib.learn.monitors.SummarySaver(tf.merge_all_summaries(),
                                                      save_steps=100,
                                                      output_dir="/tmp/test")
monitors = [validation_monitor, grad_monitor]

estimator.fit(input_fn=lambda: get_train_data(batch_size=32),
              monitors=monitors, steps=1000000)
