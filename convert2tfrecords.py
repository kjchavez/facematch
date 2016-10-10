# Creates a data pipeline for examples from the LFW dataset.
import os
import random
import cv2
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="/home/kevin/data/lfw")
parser.add_argument("--train_file", default="pairsDevTrain.txt")
parser.add_argument("--dev_file", default="pairsDevTest.txt")
parser.add_argument("--sanity_check_file", default="pairsDevSanityCheck.txt")
args = parser.parse_args()

class Example(object):
    def __init__(self, line):
        line = line.strip()
        tokens = line.split('\t')
        if len(tokens) == 3:  # positive example
            self.image1 = (tokens[0], tokens[1])
            self.image2 = (tokens[0], tokens[2])
            self.label = 1
        elif len(tokens) == 4:  # negative example
            self.image1 = (tokens[0], tokens[1])
            self.image2 = (tokens[2], tokens[3])
            self.label = 0
        else:
            print "Error: Invalid number of tokens in line:"
            print line

    def filename1(self):
        return os.path.join(args.src, "%s/%s_%04d.jpg" % (self.image1[0], self.image1[0],
                                   int(self.image1[1])))

    def filename2(self):
        return os.path.join(args.src, "%s/%s_%04d.jpg" % (self.image2[0], self.image2[0],
                                   int(self.image2[1])))

    def __str__(self):
        s = "%s:%s -- %s:%s" % (self.image1[0], self.image1[1], self.image2[0],
                              self.image2[1])
        return s

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_example(writer, example):
    image1_raw = cv2.imread(example.filename1()).tostring()
    image2_raw = cv2.imread(example.filename2()).tostring()
    example_proto = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(example.label),
        'image1_raw': _bytes_feature(image1_raw),
        'image2_raw': _bytes_feature(image2_raw)}))

    writer.write(example_proto.SerializeToString())

def convert_file(filename):
    output_filename = filename.rsplit(".", 1)[0] + ".tfrecord"
    examples = []
    with open(filename) as fp:
        num_examples = int(fp.next())
        for line in fp:
            examples.append(Example(line))

    writer = tf.python_io.TFRecordWriter(output_filename)
    random.shuffle(examples)
    for i, e in enumerate(examples):
        encode_example(writer, e)
        if i % 100 == 0:
            print "Processed %d / %d" % (i, len(examples))

    writer.close()

convert_file(args.train_file)
convert_file(args.dev_file)
convert_file(args.sanity_check_file)
