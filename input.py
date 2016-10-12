import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops

train_filename = "/home/kevin/projects/faces/pairsDevTrain.tfrecord"
dev_filename = "/home/kevin/projects/faces/pairsDevTest.tfrecord"
sanity_check_filename = "/home/kevin/projects/faces/pairsDevSanityCheck.tfrecord"

ORIG_SHAPE = (250, 250, 3)

def preprocess(image):
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                             max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                            lower=0.2, upper=1.8)

    # Should we do per-image whitening or a simple, but uniform normalization?
    return (1.0/128)*(distorted_image - 127.0)

def get_feature_input(filepattern, batch_size=1):
    filename_queue = \
        tf.train.string_input_producer(tf.matching_files(filepattern))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            "image1_raw": tf.FixedLenFeature([], tf.string),
            "image2_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })

    label = features.pop('label')

    image1 = tf.cast(tf.decode_raw(features['image1_raw'], tf.uint8),
                     tf.float32)
    image1 = preprocess(tf.reshape(image1, (250, 250, 3)))

    image2 = tf.cast(tf.decode_raw(features['image2_raw'], tf.uint8),
                     tf.float32)
    image2 = preprocess(tf.reshape(image2, (250, 250, 3)))

    # Finally, we create batches.
    image1, image2, label = tf.train.shuffle_batch([image1, image2, label],
                                                   batch_size=batch_size,
                                                   capacity=2000,
                                                   min_after_dequeue=1000,
                                                   num_threads=4)

    return {'image1': image1, 'image2': image2}, label

def get_train_data(batch_size=1):
    return get_feature_input(train_filename, batch_size=batch_size)

def get_eval_data(batch_size=1):
    return get_feature_input(dev_filename, batch_size=batch_size)

def get_sanity_check_data(batch_size=1):
    return get_feature_input(sanity_check_filename, batch_size=batch_size)
