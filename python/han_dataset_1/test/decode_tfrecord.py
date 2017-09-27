import numpy as np
import skimage.io as io
import tensorflow as tf

record_iterator = tf.python_io.tf_record_iterator('../han_dataset_train.tfrecords')

for record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(record)
  #print example.features.feature['image/encoded'].bytes_list.value[0]
  f = open('test.png', 'wb')
  f.write(example.features.feature['image/encoded'].bytes_list.value[0])
  f.close()
  print 'image/format ', example.features.feature['image/format'].bytes_list.value[0]
  print 'image/width', example.features.feature['image/width'].int64_list.value[0]
  print 'image/orig_width', example.features.feature['image/orig_width'].int64_list.value[0]
  print 'image/class', example.features.feature['image/class'].int64_list.value
  print 'max_sequence_length', len(example.features.feature['image/class'].int64_list.value)
  print 'image/unpadded_class', example.features.feature['image/unpadded_class'].int64_list.value
  s = example.features.feature['image/text'].bytes_list.value[0]
  print s
  break