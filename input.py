import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops

train_filename = "/home/kevin/projects/faces/pairsDevTrain.tfrecord"
dev_filename = "/home/kevin/projects/faces/pairsDevTest.tfrecord"

ORIG_SHAPE = (250, 250, 3)

def get_feature_columns():
    label = layers.real_valued_column("label",dtype=dtypes.int64)
    return set([label])

def preprocess(image):
    # Very simple [-1, 1] scaling
   return (1.0/128)*(image - 127.0)

def get_feature_input(filepattern, batch_size=1):
    spec = layers.create_feature_spec_for_parsing(get_feature_columns())
    spec['image1_raw'] = parsing_ops.FixedLenFeature([], tf.string)
    spec['image2_raw'] = parsing_ops.FixedLenFeature([], tf.string)

    features = tf.contrib.learn.read_batch_features(filepattern, batch_size, spec,
                                                    tf.TFRecordReader,
                                        reader_num_threads=2,
                                        parser_num_threads=2,
                                         queue_capacity=batch_size*3 + 100)
    label = features.pop('label')

    image1 = tf.cast(tf.decode_raw(features['image1_raw'], tf.uint8),
                     tf.float32)

    image2 = tf.cast(tf.decode_raw(features['image2_raw'], tf.uint8),
                     tf.float32)

    return {'image1': preprocess(image1), 'image2': preprocess(image2)}, label

def get_train_data(batch_size=1):
    return get_feature_input(train_filename, batch_size=batch_size)

def get_eval_data(batch_size=1):
    return get_feature_input(dev_filename, batch_size=batch_size)
