# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
from random import shuffle

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('samples_per_file', '50',
                    'Samples per tfrecord file')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(img_fname,
                       data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
#    img_path = os.path.join(data['folder'], data['filename'])
#    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(img_fname, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format == "PNG":
        image = image.convert('RGB')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    if data == None:
        width, height = image.size
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                img_fname.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/channels': dataset_util.int64_feature(3),
            'image/shape': dataset_util.int64_list_feature([height, width, 3]),
        }))
        return example

    else:
        width = int(data['size']['width'])
        height = int(data['size']['height'])

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        if 'object' in data:
            for obj in data['object']:
                difficult = bool(int(obj['difficult']))
                if ignore_difficult_instances and difficult:
                    continue

                difficult_obj.append(int(difficult))

                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)
                classes_text.append(obj['name'].encode('utf8'))
                classes.append(label_map_dict[obj['name']])
                truncated.append(int(obj['truncated']))
                poses.append(obj['pose'].encode('utf8'))
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature( data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        	'image/channels': dataset_util.int64_feature(3),
	    'image/shape': dataset_util.int64_list_feature([height, width, 3]),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
        return example

def get_output_filename(output_dir, idx):
#    if idx % 10 == 3:
#        return '%s/charger_test_%03d.tfrecord' % (output_dir, idx)
  #  else:
        return '%s/charger_train_%03d.tfrecord' % (output_dir, idx)

def main(_):
    data_dir = FLAGS.data_dir


    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    im_path = os.path.join(data_dir, "images")
    annotaions_path = os.path.join(data_dir, "annotations")
    idx = 0
    tf_idx = 0
    file_list = os.listdir(im_path)
    print(im_path)
    # file_list = shuffle(file_list)
    # for fname in l:
    while idx < len(file_list):
        tf_filename = get_output_filename(FLAGS.output_path, tf_idx)
        writer = tf.python_io.TFRecordWriter(tf_filename)
        j = 0
        while idx < len(file_list) and j < int(FLAGS.samples_per_file):
            fname = file_list[idx]
            img_full_path = os.path.join(im_path, fname)
            path = os.path.join(annotaions_path, fname[0:-4] + ".txt")
            data = None
            try:
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            except:
                print("No annotation file. Adding negatve sample")
            tf_example = dict_to_tf_example(img_full_path, data, data_dir, label_map_dict, FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())
            idx += 1
            j += 1
        tf_idx+=1
        writer.close()

if __name__ == '__main__':
    tf.app.run()
