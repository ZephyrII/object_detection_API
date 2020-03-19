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

import PIL.Image
import cv2
import numpy as np
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
flags.DEFINE_string('samples_per_file', '100',
                    'Samples per tfrecord file')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
# flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
#                                                           'difficult instances')
FLAGS = flags.FLAGS


#
# SETS = ['train', 'val', 'trainval', 'test']
# YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(img_fname, label_fname, class_name):
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
    # if image.format != 'JPEG':
    #     raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    label = cv2.imread(label_fname)
    width = label.shape[1]
    height = label.shape[0]
    mask_coords = np.argwhere(label == 1)

    if mask_coords.shape[0] > 0:
        rel_xmin = np.min(mask_coords[:, 1])
        rel_ymin = np.min(mask_coords[:, 0])
        rel_xmax = np.max(mask_coords[:, 1])
        rel_ymax = np.max(mask_coords[:, 0])
        xmin = [rel_xmin / width]
        ymin = [rel_ymin / height]
        xmax = [rel_xmax / width]
        ymax = [rel_ymax / height]

        classes = []
        classes_text = []
        classes_text.append(class_name.encode('utf8'))
        classes.append(1)

        feature_dict = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(img_fname.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(img_fname.encode('utf8')),
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
            # 'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            # 'image/object/truncated': dataset_util.int64_list_feature(truncated),
            # 'image/object/view': dataset_util.bytes_list_feature(poses),
        }
        encoded_mask_png_list = []
        pil_image = PIL.Image.fromarray(label)
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_mask_png_list.append(output_io.getvalue())
        # encoded_mask_png_list.append(label.tobytes())
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))
    else:
        feature_dict = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(img_fname.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(img_fname.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/channels': dataset_util.int64_feature(3),
            'image/shape': dataset_util.int64_list_feature([height, width, 3]),
        }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def get_output_filename(output_dir, idx):
    if idx % 10 == 3:
        return '%s/charger_test_%03d.tfrecord' % (output_dir, idx)
    else:
        return '%s/charger_train_%03d.tfrecord' % (output_dir, idx)


def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    # print(label_map_dict)

    im_path = os.path.join(data_dir, "images")
    labels_path = os.path.join(data_dir, "labels")
    idx = 0
    tf_idx = 0
    file_list = os.listdir(im_path)
    print(im_path)
    # file_list = shuffle(file_list)
    while idx < len(file_list):
        tf_filename = get_output_filename(FLAGS.output_path, tf_idx)
        writer = tf.python_io.TFRecordWriter(tf_filename)
        j = 0
        while j < int(FLAGS.samples_per_file):
            fname = file_list[idx]
            img_full_path = os.path.join(im_path, fname)
            label_full_path = os.path.join(labels_path, fname[:-4] + '_label' + fname[-4:])
            # print(label_full_path)
            tf_example = dict_to_tf_example(img_full_path, label_full_path, 'charger')
            writer.write(tf_example.SerializeToString())
            idx += 1
            j += 1
        tf_idx += 1
        writer.close()


if __name__ == '__main__':
    tf.app.run()
